#include <fstream>
#include "SvpSampleWk.h"
#include "SvpSampleCom.h"
#include "mpi_nnie.h"
#include "mpi_nnie_ssd.h"
#include "detectionCom.h"

using namespace std;

HI_S32 SvpSampleReadWK(const HI_CHAR *pszModelName, SVP_MEM_INFO_S *pstModelBuf)
{
    HI_S32 s32Ret = HI_FAILURE;
    HI_U32 u32Cnt = 0;
    FILE *pfModel = NULL;
    CHECK_EXP_RET(NULL == pszModelName, HI_ERR_SVP_NNIE_ILLEGAL_PARAM,
        "Error(%#x): model file name is null", HI_ERR_SVP_NNIE_ILLEGAL_PARAM);
    CHECK_EXP_RET(NULL == pstModelBuf, HI_ERR_SVP_NNIE_ILLEGAL_PARAM,
        "Error(%#x): model buf is null", HI_ERR_SVP_NNIE_NULL_PTR);

    pfModel = SvpSampleOpenFile(pszModelName, "rb");
    CHECK_EXP_RET(NULL == pfModel, HI_ERR_SVP_NNIE_OPEN_FILE,
        "Error(%#x): open model file(%s) failed", HI_ERR_SVP_NNIE_OPEN_FILE, pszModelName);

    fseek(pfModel, 0, SEEK_END);
    pstModelBuf->u32Size = ftell(pfModel);
    fseek(pfModel, 0, SEEK_SET);

    s32Ret = SvpSampleMallocMem(NULL, NULL, pstModelBuf->u32Size, pstModelBuf);
    CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail, "Error(%#x): Malloc model buf failed!", s32Ret);

    u32Cnt = (HI_U32)fread((void*)pstModelBuf->u64VirAddr, pstModelBuf->u32Size, 1, pfModel);
    if (1 != u32Cnt)
    {
        s32Ret = HI_FAILURE;
    }

Fail:
    SvpSampleCloseFile(pfModel);

    return s32Ret;
}

static HI_S32 SvpSampleOneSegCommonInit(SVP_NNIE_CFG_S *pstClfCfg, SVP_NNIE_ONE_SEG_S *pstComfParam, SVP_SAMPLE_LSTMRunTimeCtx *pstLSTMCtx)
{
    HI_S32 s32Ret = HI_SUCCESS;
    HI_U32 i = 0;
    HI_U16 u16SrcNum = 0, u16DstNum = 0;
    HI_U32 u32SegCnt = 0;

    HI_U32 u32Num = 0;
    HI_U32 u32MaxClfNum = 0;

    HI_CHAR aszImg[SVP_SAMPLE_MAX_PATH] = { '\0' };

    SVP_NNIE_NODE_S *pstSrcNode = NULL;
    SVP_NNIE_NODE_S *pstDstNode = NULL;
    SVP_BLOB_TYPE_E enType;

    SVP_MEM_INFO_S *pstModelBuf = &pstComfParam->stModelBuf;
    SVP_MEM_INFO_S *pstTmpBuf = &pstComfParam->stTmpBuf;
    SVP_MEM_INFO_S *pstTskBuf = &pstComfParam->stTskBuf;

    /******************** step1, load wk file, *******************************/
    s32Ret = SvpSampleReadWK(pstClfCfg->pszModelName, pstModelBuf);
    CHECK_EXP_RET(HI_SUCCESS != s32Ret, s32Ret, "Error(%#x): read model file(%s) failed", s32Ret, pstClfCfg->pszModelName);

    s32Ret = HI_MPI_SVP_NNIE_LoadModel(pstModelBuf, &(pstComfParam->stModel));
    CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail1, "Error(%#x): LoadModel from %s failed!", s32Ret, pstClfCfg->pszModelName);

    pstComfParam->u32TmpBufSize = pstComfParam->stModel.u32TmpBufSize;

    /******************** step2, malloc tmp_buf *******************************/
    s32Ret = SvpSampleMallocMem(NULL, NULL, pstComfParam->u32TmpBufSize, pstTmpBuf);
    CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail2, "Error(%#x): Malloc tmp buf failed!", s32Ret);

    /******************** step3, get tsk_buf size *******************************/
    CHECK_EXP_GOTO(pstComfParam->stModel.u32NetSegNum != 1, Fail3, "netSegNum should be 1");
    s32Ret = HI_MPI_SVP_NNIE_GetTskBufSize(pstClfCfg->u32MaxInputNum, pstClfCfg->u32MaxBboxNum,
        &pstComfParam->stModel, &pstComfParam->u32TaskBufSize, pstComfParam->stModel.u32NetSegNum);
    CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail3, "Error(%#x): GetTaskSize failed!", s32Ret);

    /******************** step4, malloc tsk_buf size *******************************/
    s32Ret = SvpSampleMallocMem(NULL, NULL, pstComfParam->u32TaskBufSize, pstTskBuf);
    CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail3, "Error(%#x): Malloc task buf failed!", s32Ret);

    /*********** step5, check and open all input images list file ******************/
    // all input pic_list file should have the same num of input image
    u32Num = 0;
    CHECK_EXP_GOTO(!pstClfCfg->paszPicList[0], Fail4, "Error(%#x): input pic_list[0] is null",HI_ERR_SVP_NNIE_ILLEGAL_PARAM);
    pstComfParam->fpSrc[0] = SvpSampleOpenFile(pstClfCfg->paszPicList[0], "r");
    CHECK_EXP_GOTO(!(pstComfParam->fpSrc[0]), Fail4, "Error(%#x), Open file(%s) failed!",
        HI_ERR_SVP_NNIE_OPEN_FILE, pstClfCfg->paszPicList[0]);
    while (fgets(aszImg, SVP_SAMPLE_MAX_PATH, pstComfParam->fpSrc[0]) != NULL)
    {
        u32Num++;
    }
    pstComfParam->u32TotalImgNum = u32Num;

    u16SrcNum = pstComfParam->stModel.astSeg[0].u16SrcNum;
    for (i = 1; i < u16SrcNum; i++)
    {
        u32Num = 0;
        CHECK_EXP_GOTO(!pstClfCfg->paszPicList[i], Fail5,
            "u16SrcNum = %d, but the %dth input pic_list file is null", u16SrcNum, i);
        pstComfParam->fpSrc[i] = SvpSampleOpenFile(pstClfCfg->paszPicList[i], "r");
        CHECK_EXP_GOTO(!(pstComfParam->fpSrc[i]), Fail5, "Error(%#x), Open file(%s) failed!",
            HI_ERR_SVP_NNIE_OPEN_FILE, pstClfCfg->paszPicList[i]);
        while (fgets(aszImg, SVP_SAMPLE_MAX_PATH, pstComfParam->fpSrc[i]) != NULL)
        {
            u32Num++;
        }
        CHECK_EXP_GOTO(u32Num != pstComfParam->u32TotalImgNum, Fail5, 
            "The %dth pic_list file has a num of %d, which is not equal to %d",
            i, u32Num, pstComfParam->u32TotalImgNum);
    }

    /*********** step6, if need label then open all label file ******************/
    u16DstNum = pstComfParam->stModel.astSeg[0].u16DstNum;
    if (pstClfCfg->bNeedLabel)
    {
        // all input label file should have the same num of labels of input image
        for (i = 0; i < u16DstNum; i++)
        {
            u32Num = 0;
            CHECK_EXP_GOTO(!pstClfCfg->paszLabel[i], Fail5,
                "u16DstNum = %d, but the %dth input label file is null", u16DstNum, i);
            pstComfParam->fpLabel[i] = SvpSampleOpenFile(pstClfCfg->paszLabel[i], "r");
            CHECK_EXP_GOTO(!(pstComfParam->fpLabel[i]), Fail5, "Error(%#x), Open file(%s) failed!",
                HI_ERR_SVP_NNIE_OPEN_FILE, pstClfCfg->paszLabel[i]);
            while (fgets(aszImg, SVP_SAMPLE_MAX_PATH, pstComfParam->fpLabel[i]) != NULL)
            {
                u32Num++;
            }
        }
        CHECK_EXP_GOTO(u32Num != pstComfParam->u32TotalImgNum, Fail5,
                "The %dth label file has a num of %d, which is not equal to %d",
                i, u32Num, pstComfParam->u32TotalImgNum);
    }

    /*********** step7, malloc memory of src blob, dst blob and post-process mem ***********/
    u32Num = SVP_SAMPLE_MIN(pstComfParam->u32TotalImgNum, pstClfCfg->u32MaxInputNum);
    // malloc src, dst blob buf
    for (u32SegCnt = 0; u32SegCnt < pstComfParam->stModel.u32NetSegNum; ++u32SegCnt)
    {
        HI_U32 u32SrcW, u32SrcH, u32SrcC, u32DstW, u32DstH, u32DstC, u32Dim;
        pstSrcNode = (SVP_NNIE_NODE_S *)(pstComfParam->stModel.astSeg[u32SegCnt].astSrcNode);
        pstDstNode = (SVP_NNIE_NODE_S *)(pstComfParam->stModel.astSeg[u32SegCnt].astDstNode);

        // malloc src blob buf;
        for (i = 0; i < pstComfParam->stModel.astSeg[u32SegCnt].u16SrcNum; ++i)
        {
            enType = pstSrcNode->enType;
            if (SVP_BLOB_TYPE_SEQ_S32 == enType)
            {
                u32Dim = pstSrcNode->unShape.u32Dim;
                s32Ret = SvpSampleMallocSeqBlob(&pstComfParam->astSrc[i], enType, u32Num, u32Dim, pstLSTMCtx);
            }
            else
            {
                u32SrcC = pstSrcNode->unShape.stWhc.u32Chn;
                u32SrcW = pstSrcNode->unShape.stWhc.u32Width;
                u32SrcH = pstSrcNode->unShape.stWhc.u32Height;
                s32Ret = SvpSampleMallocBlob(&pstComfParam->astSrc[i],
                    enType, u32Num, u32SrcC, u32SrcW, u32SrcH);
            }
            CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail6, "Error(%#x): Malloc src blob failed!", s32Ret);
            ++pstSrcNode;
        }

        // malloc dst blob buf;
        for (i = 0; i < pstComfParam->stModel.astSeg[u32SegCnt].u16DstNum; ++i)
        {
            enType = pstDstNode->enType;
            if (SVP_BLOB_TYPE_SEQ_S32 == enType)
            {
                u32Dim = pstDstNode->unShape.u32Dim;
                s32Ret = SvpSampleMallocSeqBlob(&pstComfParam->astDst[i], enType, u32Num, u32Dim, pstLSTMCtx);
            }
            else
            {
                u32DstC = pstDstNode->unShape.stWhc.u32Chn;
                u32DstW = pstDstNode->unShape.stWhc.u32Width;
                u32DstH = pstDstNode->unShape.stWhc.u32Height;

                s32Ret = SvpSampleMallocBlob(&pstComfParam->astDst[i],
                    enType, u32Num, u32DstC, u32DstW, u32DstH);
            }
            CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail6, "Error(%#x): Malloc dst blob failed!", s32Ret);

            // normal classification net which has FC layer before the last softmax layer
            if (pstComfParam->astDst[i].enType == SVP_BLOB_TYPE_VEC_S32)
            {
                pstComfParam->au32ClfNum[i] = pstComfParam->astDst[i].unShape.stWhc.u32Width;
            }
            // classification net, such as squeezenet, which has global_pooling layer before the last softmax layer
            else
            {
                pstComfParam->au32ClfNum[i] = pstComfParam->astDst[i].unShape.stWhc.u32Chn;
            }
            if (u32MaxClfNum < pstComfParam->astDst[i].unShape.stWhc.u32Width)
            {
                u32MaxClfNum = pstComfParam->au32ClfNum[i];
            }

            ++pstDstNode;
        }

        // memory need by post-process of getting top-N
        if (pstClfCfg->bNeedLabel)
        {
            pstComfParam->pstMaxClfIdScore = (SVP_SAMPLE_CLF_RES_S*)malloc(u32MaxClfNum * sizeof(SVP_SAMPLE_CLF_RES_S));
            CHECK_EXP_GOTO(!pstComfParam->pstMaxClfIdScore, Fail6, "Error: Malloc pstMaxclfIdScore failed!");
            for (i = 0; i < pstComfParam->stModel.astSeg[u32SegCnt].u16DstNum; ++i)
            {
                pstComfParam->pastClfRes[i] = (SVP_SAMPLE_CLF_RES_S*)malloc(pstComfParam->au32ClfNum[i] * sizeof(SVP_SAMPLE_CLF_RES_S));
                CHECK_EXP_GOTO(!pstComfParam->pastClfRes[i], Fail6, "Error: Malloc pastClfRes[%d] failed!", i);
            }
        }
        else
        {
            pstComfParam->fpLabel[0] = NULL;
        }
    }

    /************************** step8, set ctrl param **************************/
    pstComfParam->stCtrl.enNnieId    = SVP_NNIE_ID_0;
    pstComfParam->stCtrl.u32NetSegId = 0;
    pstComfParam->stCtrl.u32SrcNum = pstComfParam->stModel.astSeg[0].u16SrcNum;
    pstComfParam->stCtrl.u32DstNum = pstComfParam->stModel.astSeg[0].u16DstNum;
    memcpy(&pstComfParam->stCtrl.stTmpBuf, &pstComfParam->stTmpBuf, sizeof(SVP_MEM_INFO_S));
    memcpy(&pstComfParam->stCtrl.stTskBuf, &pstComfParam->stTskBuf, sizeof(SVP_MEM_INFO_S));

    return s32Ret;

Fail6:
    if (!pstComfParam->pstMaxClfIdScore)
    {
        free(pstComfParam->pstMaxClfIdScore);
        pstComfParam->pstMaxClfIdScore = NULL;
    }
    for (i = 0; i < u16DstNum; ++i)
    {
        SvpSampleFreeBlob(&pstComfParam->astDst[i]);
        if (!pstComfParam->pastClfRes[i])
        {
            free(pstComfParam->pastClfRes[i]);
            pstComfParam->pastClfRes[i] = NULL;
        }
    }
    for (i = 0; i < u16SrcNum; ++i)
    {
        SvpSampleFreeBlob(&pstComfParam->astSrc[i]);
    }
Fail5:
    for (i = 0; i < u16SrcNum; ++i)
    {
        if (!pstComfParam->fpSrc[i])
        {
            fclose(pstComfParam->fpSrc[i]);
            pstComfParam->fpSrc[i] = NULL;
        }
    }
    for (i = 0;i < u16DstNum; ++i)
    {
        if (!pstComfParam->fpLabel[i])
        {
            fclose(pstComfParam->fpLabel[i]);
            pstComfParam->fpLabel[i] = NULL;
        }
    }
Fail4:
    SvpSampleMemFree(&pstComfParam->stTskBuf);
Fail3:
    SvpSampleMemFree(&pstComfParam->stTmpBuf);
Fail2:
    HI_MPI_SVP_NNIE_UnloadModel(&(pstComfParam->stModel));
Fail1:
    SvpSampleMemFree(&pstComfParam->stModelBuf);

    return s32Ret;
}

HI_S32 SvpSampleOneSegCnnInit(SVP_NNIE_CFG_S *pstClfCfg, SVP_NNIE_ONE_SEG_S *pstComParam)
{
    return SvpSampleOneSegCommonInit(pstClfCfg, pstComParam, NULL);
}

HI_S32 SvpSampleLSTMInit(SVP_NNIE_CFG_S *pstClfCfg, SVP_NNIE_ONE_SEG_S *pstComParam, SVP_SAMPLE_LSTMRunTimeCtx *pstLSTMCtx)
{
    return SvpSampleOneSegCommonInit(pstClfCfg, pstComParam, pstLSTMCtx);
}

static void SvpSampleOneSegCommDeinit(SVP_NNIE_ONE_SEG_S *pstComParam)
{
    HI_U32 i, j;
    if (!pstComParam)
    {
        printf("pstComParma is NULL\n");
        return;
    }

    for (i = 0; i < pstComParam->stModel.u32NetSegNum; ++i)
    {
        for (j = 0; j < pstComParam->stModel.astSeg[i].u16DstNum; ++j)
        {
            SvpSampleFreeBlob(&pstComParam->astDst[j]);

            if (pstComParam->pastClfRes[j])
            {
                free(pstComParam->pastClfRes[j]);
                pstComParam->pastClfRes[j] = NULL;
            }
        }

        for (j = 0; j < pstComParam->stModel.astSeg[i].u16SrcNum; ++j)
        {
            SvpSampleFreeBlob(&pstComParam->astSrc[j]);
            SvpSampleCloseFile(pstComParam->fpSrc[j]);
            SvpSampleCloseFile(pstComParam->fpLabel[j]);
        }
    }

    if (pstComParam->pstMaxClfIdScore)
    {
        free(pstComParam->pstMaxClfIdScore);
        pstComParam->pstMaxClfIdScore = NULL;
    }

    SvpSampleMemFree(&pstComParam->stTskBuf);
    SvpSampleMemFree(&pstComParam->stTmpBuf);
    HI_MPI_SVP_NNIE_UnloadModel(&(pstComParam->stModel));
    SvpSampleMemFree(&pstComParam->stModelBuf);

    memset(pstComParam, 0, sizeof(SVP_NNIE_ONE_SEG_S));
}

void SvpSampleOneSegCnnDeinit(SVP_NNIE_ONE_SEG_S *pstComParam)
{
    SvpSampleOneSegCommDeinit(pstComParam);
}

HI_S32 SvpSampleLSTMDeinit(SVP_NNIE_ONE_SEG_S *pstComParam)
{
    SvpSampleOneSegCommDeinit(pstComParam);
    return HI_SUCCESS;
}


/*********************************/
//一阶段检测卷积网络初始化
HI_S32 SvpSampleOneSegDetCnnInit(SVP_NNIE_CFG_S *pstClfCfg, SVP_NNIE_ONE_SEG_DET_S *pstComfParam, const HI_U8 netType)
{
    HI_S32 s32Ret = HI_SUCCESS;
    HI_U32 i = 0;
    HI_U16 u16SrcNum = 0;
    HI_U32 u32SegCnt = 0;

    HI_U32 u32Num = 0;
    
    HI_CHAR aszImg[SVP_SAMPLE_MAX_PATH] = { '\0' };

    SVP_NNIE_NODE_S *pstSrcNode = NULL;
    SVP_NNIE_NODE_S *pstDstNode = NULL;
    SVP_BLOB_TYPE_E enType;

    SVP_MEM_INFO_S *pstModelBuf = &pstComfParam->stModelBuf;
    SVP_MEM_INFO_S *pstTmpBuf = &pstComfParam->stTmpBuf;
    SVP_MEM_INFO_S *pstTskBuf = &pstComfParam->stTskBuf;

    /******************** step1, load wk file, *******************************/
    s32Ret = SvpSampleReadWK(pstClfCfg->pszModelName, pstModelBuf);
    CHECK_EXP_RET(HI_SUCCESS != s32Ret, s32Ret, "Error(%#x): read model file(%s) failed", s32Ret, pstClfCfg->pszModelName);

    s32Ret = HI_MPI_SVP_NNIE_LoadModel(pstModelBuf, &(pstComfParam->stModel));
    CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail1, "Error(%#x): LoadModel from %s failed!", s32Ret, pstClfCfg->pszModelName);

    pstComfParam->u32TmpBufSize = pstComfParam->stModel.u32TmpBufSize;

    /******************** step2, malloc tmp_buf *******************************/
    s32Ret = SvpSampleMallocMem(NULL, NULL, pstComfParam->u32TmpBufSize, pstTmpBuf);
    CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail2, "Error(%#x): Malloc tmp buf failed!", s32Ret);

    /******************** step3, get tsk_buf size *******************************/
    CHECK_EXP_GOTO(pstComfParam->stModel.u32NetSegNum != 1, Fail3, "netSegNum should be 1");
    s32Ret = HI_MPI_SVP_NNIE_GetTskBufSize(pstClfCfg->u32MaxInputNum, pstClfCfg->u32MaxBboxNum,
        &pstComfParam->stModel, &pstComfParam->u32TaskBufSize, pstComfParam->stModel.u32NetSegNum);
    CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail3, "Error(%#x): GetTaskSize failed!", s32Ret);

    /******************** step4, malloc tsk_buf size *******************************/
    s32Ret = SvpSampleMallocMem(NULL, NULL, pstComfParam->u32TaskBufSize, pstTskBuf);
    CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail3, "Error(%#x): Malloc task buf failed!", s32Ret);

    /*********** step5, check and open all input images list file ******************/
    // all input pic_list file should have the same num of input image
    u32Num = 0;
    CHECK_EXP_GOTO(!pstClfCfg->paszPicList[0], Fail4, "Error(%#x): input pic_list[0] is null", HI_ERR_SVP_NNIE_ILLEGAL_PARAM);
    pstComfParam->fpSrc[0] = SvpSampleOpenFile(pstClfCfg->paszPicList[0], "r");
    CHECK_EXP_GOTO(!(pstComfParam->fpSrc[0]), Fail4, "Error(%#x), Open file(%s) failed!",
        HI_ERR_SVP_NNIE_OPEN_FILE, pstClfCfg->paszPicList[0]);
    while (fgets(aszImg, SVP_SAMPLE_MAX_PATH, pstComfParam->fpSrc[0]) != NULL)
    {
        u32Num++;
    }
    pstComfParam->u32TotalImgNum = u32Num;

    u16SrcNum = pstComfParam->stModel.astSeg[0].u16SrcNum;
    for (i = 1; i < u16SrcNum; i++)
    {
        u32Num = 0;
        CHECK_EXP_GOTO(!pstClfCfg->paszPicList[i], Fail5,
            "u16SrcNum = %d, but the %dth input pic_list file is null", u16SrcNum, i);
        pstComfParam->fpSrc[i] = SvpSampleOpenFile(pstClfCfg->paszPicList[i], "r");
        CHECK_EXP_GOTO(!(pstComfParam->fpSrc[i]), Fail5, "Error(%#x), Open file(%s) failed!",
            HI_ERR_SVP_NNIE_OPEN_FILE, pstClfCfg->paszPicList[i]);
        while (fgets(aszImg, SVP_SAMPLE_MAX_PATH, pstComfParam->fpSrc[i]) != NULL)
        {
            u32Num++;
        }
        CHECK_EXP_GOTO(u32Num != pstComfParam->u32TotalImgNum, Fail5,
            "The %dth pic_list file has a num of %d, which is not equal to %d",
            i, u32Num, pstComfParam->u32TotalImgNum);
    }

    /*********** step6, malloc memory of src blob, dst blob and post-process mem ***********/
    u32Num = SVP_SAMPLE_MIN(pstComfParam->u32TotalImgNum, pstClfCfg->u32MaxInputNum);
    // malloc src, dst blob buf
    for (u32SegCnt = 0; u32SegCnt < pstComfParam->stModel.u32NetSegNum; ++u32SegCnt)
    {
        HI_U32 u32SrcW, u32SrcH, u32SrcC, u32DstW, u32DstH, u32DstC;
        pstSrcNode = (SVP_NNIE_NODE_S *)(pstComfParam->stModel.astSeg[u32SegCnt].astSrcNode);
        pstDstNode = (SVP_NNIE_NODE_S *)(pstComfParam->stModel.astSeg[u32SegCnt].astDstNode);

        // malloc src blob buf;
        for (i = 0; i < pstComfParam->stModel.astSeg[u32SegCnt].u16SrcNum; ++i)
        {
            enType = pstSrcNode->enType;
            u32SrcC = pstSrcNode->unShape.stWhc.u32Chn;
            u32SrcW = pstSrcNode->unShape.stWhc.u32Width;
            u32SrcH = pstSrcNode->unShape.stWhc.u32Height;
            s32Ret = SvpSampleMallocBlob(&pstComfParam->astSrc[i],
                enType, u32Num, u32SrcC, u32SrcW, u32SrcH);
            CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail6, "Error(%#x): Malloc src blob failed!", s32Ret);
            ++pstSrcNode;
        }

        // malloc dst blob buf;
        for (i = 0; i < pstComfParam->stModel.astSeg[u32SegCnt].u16DstNum; ++i)
        {
            enType = pstDstNode->enType;
            u32DstC = pstDstNode->unShape.stWhc.u32Chn;
            u32DstW = pstDstNode->unShape.stWhc.u32Width;
            u32DstH = pstDstNode->unShape.stWhc.u32Height;

            s32Ret = SvpSampleMallocBlob(&pstComfParam->astDst[i],
                enType, u32Num, u32DstC, u32DstW, u32DstH);
            CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail6, "Error(%#x): Malloc dst blob failed!", s32Ret);
            ++pstDstNode;
        }
    }

    /************************** step8, set ctrl param **************************/
    pstComfParam->stCtrl.enNnieId = SVP_NNIE_ID_0;
    pstComfParam->stCtrl.u32NetSegId = 0;
    pstComfParam->stCtrl.u32SrcNum = pstComfParam->stModel.astSeg[0].u16SrcNum;
    pstComfParam->stCtrl.u32DstNum = pstComfParam->stModel.astSeg[0].u16DstNum;
    memcpy(&pstComfParam->stCtrl.stTmpBuf, &pstComfParam->stTmpBuf, sizeof(SVP_MEM_INFO_S));
    memcpy(&pstComfParam->stCtrl.stTskBuf, &pstComfParam->stTskBuf, sizeof(SVP_MEM_INFO_S));


    return s32Ret;

Fail6:
    for (i = 0; i < u16SrcNum; ++i)
    {
        SvpSampleFreeBlob(&pstComfParam->astSrc[i]);
    }
Fail5:
    for (i = 0; i < u16SrcNum; ++i)
    {
        if (!pstComfParam->fpSrc[i])
        {
            fclose(pstComfParam->fpSrc[i]);
            pstComfParam->fpSrc[i] = NULL;
        }
    }
Fail4:
    SvpSampleMemFree(&pstComfParam->stTskBuf);
Fail3:
    SvpSampleMemFree(&pstComfParam->stTmpBuf);
Fail2:
    HI_MPI_SVP_NNIE_UnloadModel(&(pstComfParam->stModel));
Fail1:
    SvpSampleMemFree(&pstComfParam->stModelBuf);

    return s32Ret;
}

//一阶段检测卷积网络内存注销，参数：一阶段检测参数结构
void SvpSampleOneSegDetCnnDeinit(SVP_NNIE_ONE_SEG_DET_S *pstComParam)
{
    HI_U32 i, j;
    if (!pstComParam)
    {
        printf("pstComParma is NULL\n");
        return;
    }

    for (i = 0; i < pstComParam->stModel.u32NetSegNum; ++i)
    {
        for (j = 0; j < pstComParam->stModel.astSeg[i].u16DstNum; ++j)
        {
            SvpSampleFreeBlob(&pstComParam->astDst[j]);
        }

        for (j = 0; j < pstComParam->stModel.astSeg[i].u16SrcNum; ++j)
        {
            SvpSampleFreeBlob(&pstComParam->astSrc[j]);
            SvpSampleCloseFile(pstComParam->fpSrc[j]);
        }
    }

    SvpSampleMemFree(&pstComParam->stTskBuf);
    SvpSampleMemFree(&pstComParam->stTmpBuf);
    HI_MPI_SVP_NNIE_UnloadModel(&(pstComParam->stModel));
    SvpSampleMemFree(&pstComParam->stModelBuf);

    memset(pstComParam, 0, sizeof(SVP_NNIE_ONE_SEG_DET_S));
}

//多阶段卷积网络的初始化，参数配置文件，参数结构体，src和dst的align数值
HI_S32 SvpSampleMultiSegCnnInit(SVP_NNIE_CFG_S *pstComCfg, SVP_NNIE_MULTI_SEG_S *pstComfParam,
    HI_U32 *pu32SrcAlign, HI_U32 *pu32DstAlign)
{
    HI_S32 s32Ret = HI_SUCCESS;//返回状态
    HI_U32 i = 0;
    HI_U16 u16SrcNum = 0;//输入个数
    HI_U32 u32SegCnt = 0;//网络段数
    HI_U32 u32MaxTaskSize = 0;
    HI_U32 u32CtrlCnt = 0, u32BboxCtrlCnt = 0;
    HI_U32 u32DstCnt = 0, u32SrcCnt = 0, u32RPNCnt = 0;//输入计数，输出计数，RPN计数

    HI_U32 u32Num = 0;
    HI_U32 u32NumWithBbox = 0;

    HI_CHAR aszImg[SVP_SAMPLE_MAX_PATH] = { '\0' };

    SVP_NNIE_NODE_S *pstSrcNode = NULL;//输入节点
    SVP_NNIE_NODE_S *pstDstNode = NULL;//输出节点
    SVP_BLOB_TYPE_E enType;

    SVP_MEM_INFO_S *pstModelBuf = &pstComfParam->stModelBuf;//模型存储
    SVP_MEM_INFO_S *pstTmpBuf = &pstComfParam->stTmpBuf;//缓存指针
    SVP_MEM_INFO_S *pstTskBuf = &pstComfParam->astTskBuf[0];//任务存储

    /******************** step1, load wk file, *******************************/
    //读取模型到内存区域，参数配置文件中拿模型名称，加载到相应的内存区域
    s32Ret = SvpSampleReadWK(pstComCfg->pszModelName, pstModelBuf);
    CHECK_EXP_RET(HI_SUCCESS != s32Ret, s32Ret, "Error(%#x): read model file(%s) failed", s32Ret, pstComCfg->pszModelName);

    //加载模型到nnie框架，参数：内存信息，nnie框架模型
    s32Ret = HI_MPI_SVP_NNIE_LoadModel(pstModelBuf, &(pstComfParam->stModel));
    CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail1, "Error(%#x): LoadModel from %s failed!", s32Ret, pstComCfg->pszModelName);

    //这个bufsize有多处定义么？
    pstComfParam->u32TmpBufSize = pstComfParam->stModel.u32TmpBufSize;

    /******************** step2, malloc tmp_buf *******************************/
    //申请缓存，参数：内存大小从上一步得到，地址由指针pstTmpBuf指向
    s32Ret = SvpSampleMallocMem(NULL, NULL, pstComfParam->u32TmpBufSize, pstTmpBuf);
    CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail2, "Error(%#x): Malloc tmp buf failed!", s32Ret);

    /******************** step3, get task_buf size *******************************/
    //计算任务存储
    CHECK_EXP_GOTO(pstComfParam->stModel.u32NetSegNum < 2, Fail3, "netSegNum should be larger than 1");///delete it???

    s32Ret = HI_MPI_SVP_NNIE_GetTskBufSize(pstComCfg->u32MaxInputNum, pstComCfg->u32MaxBboxNum,
        &pstComfParam->stModel, pstComfParam->au32TaskBufSize, pstComfParam->stModel.u32NetSegNum);
    CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail3, "Error(%#x): GetTaskSize failed!", s32Ret);

    /******************** step4, malloc tsk_buf size *******************************/
    //NNIE and CPU running at interval. get max task_buf size
    //nnie框架和cpu交替运行，申请最大的任务内存
    u32MaxTaskSize = pstComfParam->au32TaskBufSize[0];
    for (i = 1; i<pstComfParam->stModel.u32NetSegNum; i++)
    {
        if (u32MaxTaskSize < pstComfParam->au32TaskBufSize[i])
        {
            u32MaxTaskSize = pstComfParam->au32TaskBufSize[i];
        }
    }

    s32Ret = SvpSampleMallocMem(NULL, NULL, u32MaxTaskSize, pstTskBuf);
    CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail3, "Error(%#x): Malloc task buf failed!", s32Ret);

    /*********** step5, check and open all input images list file ******************/
    // all input pic_list file should have the same num of input image
    for (i = 0; i < SVP_NNIE_MAX_INPUT_NUM; i++)
    {
        pstComfParam->fpSrc[i] = NULL;
    }
    u32Num = 0;
    CHECK_EXP_GOTO(!pstComCfg->paszPicList[0], Fail4, "Error(%#x): input pic_list[0] is null", HI_ERR_SVP_NNIE_ILLEGAL_PARAM);
    pstComfParam->fpSrc[0] = SvpSampleOpenFile(pstComCfg->paszPicList[0], "r");
    CHECK_EXP_GOTO(!(pstComfParam->fpSrc[0]), Fail4, "Error(%#x), Open file(%s) failed!",
        HI_ERR_SVP_NNIE_OPEN_FILE, pstComCfg->paszPicList[0]);
    while (fgets(aszImg, SVP_SAMPLE_MAX_PATH, pstComfParam->fpSrc[0]) != NULL)
    {
        u32Num++;
    }
    pstComfParam->u32TotalImgNum = u32Num;

    CHECK_EXP_GOTO(pstComfParam->u32TotalImgNum != 1, Fail5,
        "u32TotalImgNum = %d, but the MultiSeg network only supports single frame input", u32Num);

    u16SrcNum = pstComfParam->stModel.astSeg[0].u16SrcNum;
    for (i = 1; i < u16SrcNum; i++)
    {
        u32Num = 0;
        CHECK_EXP_GOTO(!pstComCfg->paszPicList[i], Fail5,
            "u16SrcNum = %d, but the %dth input pic_list file is null", u16SrcNum, i);
        pstComfParam->fpSrc[i] = SvpSampleOpenFile(pstComCfg->paszPicList[i], "r");
        CHECK_EXP_GOTO(!(pstComfParam->fpSrc[i]), Fail5, "Error(%#x), Open file(%s) failed!",
            HI_ERR_SVP_NNIE_OPEN_FILE, pstComCfg->paszPicList[i]);
        while (fgets(aszImg, SVP_SAMPLE_MAX_PATH, pstComfParam->fpSrc[i]) != NULL)
        {
            u32Num++;
        }
        CHECK_EXP_GOTO(u32Num != pstComfParam->u32TotalImgNum, Fail5,
            "The %dth pic_list file has a num of %d, which is not equal to %d",
            i, u32Num, pstComfParam->u32TotalImgNum);
    }

     /*********** step6, malloc memory of src blob, dst blob and post-process mem ***********/
    u32Num = SVP_SAMPLE_MIN(pstComfParam->u32TotalImgNum, pstComCfg->u32MaxInputNum);
    // malloc src, dst blob buf
    for (u32SegCnt = 0; u32SegCnt < pstComfParam->stModel.u32NetSegNum; ++u32SegCnt)
    {
        HI_U32 u32SrcW, u32SrcH, u32SrcC, u32DstW, u32DstH, u32DstC;
        pstSrcNode = (SVP_NNIE_NODE_S *)(pstComfParam->stModel.astSeg[u32SegCnt].astSrcNode);
        pstDstNode = (SVP_NNIE_NODE_S *)(pstComfParam->stModel.astSeg[u32SegCnt].astDstNode);

        // malloc src blob buf;
        for (i = 0; i < pstComfParam->stModel.astSeg[u32SegCnt].u16SrcNum; ++i)
        {
            enType = pstSrcNode->enType;
            u32SrcC = pstSrcNode->unShape.stWhc.u32Chn;
            u32SrcW = pstSrcNode->unShape.stWhc.u32Width;
            u32SrcH = pstSrcNode->unShape.stWhc.u32Height;
            s32Ret = SvpSampleMallocBlob(&pstComfParam->astSrc[i + u32SrcCnt],
                enType, u32Num, u32SrcC, u32SrcW, u32SrcH, pu32SrcAlign ? pu32SrcAlign[i] : STRIDE_ALIGN);
            CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail6, "Error(%#x): Malloc src blob failed!", s32Ret);

            ++pstSrcNode;
        }
        u32SrcCnt += pstComfParam->stModel.astSeg[u32SegCnt].u16SrcNum;

        // malloc dst blob buf;
        for (i = 0; i < pstComfParam->stModel.astSeg[u32SegCnt].u16DstNum; ++i)
        {
            enType = pstDstNode->enType;
            u32DstC = pstDstNode->unShape.stWhc.u32Chn;
            u32DstW = pstDstNode->unShape.stWhc.u32Width;
            u32DstH = pstDstNode->unShape.stWhc.u32Height;

            u32NumWithBbox = (pstComCfg->u32MaxBboxNum > 0 && pstComfParam->stModel.astSeg[u32SegCnt].u16RoiPoolNum > 0) ?
                u32Num*pstComCfg->u32MaxBboxNum : u32Num;
            s32Ret = SvpSampleMallocBlob(&pstComfParam->astDst[i + u32DstCnt],
                enType, u32NumWithBbox, u32DstC, u32DstW, u32DstH, pu32DstAlign ? pu32DstAlign[i] : STRIDE_ALIGN);
            CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail6, "Error(%#x): Malloc dst blob failed!", s32Ret);

            ++pstDstNode;
        }
        u32DstCnt += pstComfParam->stModel.astSeg[u32SegCnt].u16DstNum;

        //malloc RPN blob buf if exists
        if (pstComCfg->u32MaxBboxNum > 0)
        {
            for (i = 0; i < pstComfParam->stModel.astSeg[u32SegCnt].u16RoiPoolNum; ++i)
            {
                s32Ret = SvpSampleMallocRPNBlob(&pstComfParam->stRPN[u32RPNCnt + i], pstComCfg->u32MaxBboxNum);
                CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail7, "Error(%#x): Malloc rpn blob failed!", s32Ret)
            }
            u32RPNCnt += pstComfParam->stModel.astSeg[u32SegCnt].u16RoiPoolNum;
        }
    }

    /************************** step7, set ctrl param **************************/
    for (u32SegCnt = 0; u32SegCnt < pstComfParam->stModel.u32NetSegNum; ++u32SegCnt)
    {
        if (SVP_NNIE_NET_TYPE_ROI == pstComfParam->stModel.astSeg[u32SegCnt].enNetType)
        {
            pstComfParam->astBboxCtrl[u32BboxCtrlCnt].enNnieId = SVP_NNIE_ID_0;
            pstComfParam->astBboxCtrl[u32BboxCtrlCnt].u32NetSegId = u32SegCnt;
            pstComfParam->astBboxCtrl[u32BboxCtrlCnt].u32ProposalNum = 1;
            pstComfParam->astBboxCtrl[u32BboxCtrlCnt].u32SrcNum = pstComfParam->stModel.astSeg[u32SegCnt].u16SrcNum;
            pstComfParam->astBboxCtrl[u32BboxCtrlCnt].u32DstNum = pstComfParam->stModel.astSeg[u32SegCnt].u16DstNum;
            memcpy(&pstComfParam->astBboxCtrl[u32BboxCtrlCnt].stTmpBuf, &pstComfParam->stTmpBuf, sizeof(SVP_MEM_INFO_S));
            memcpy(&pstComfParam->astBboxCtrl[u32BboxCtrlCnt].stTskBuf, &pstComfParam->astTskBuf[0], sizeof(SVP_MEM_INFO_S));
            u32BboxCtrlCnt++;
        }
        else
        {
            pstComfParam->astCtrl[u32CtrlCnt].enNnieId = SVP_NNIE_ID_0;
            pstComfParam->astCtrl[u32CtrlCnt].u32NetSegId = u32SegCnt;
            pstComfParam->astCtrl[u32CtrlCnt].u32SrcNum = pstComfParam->stModel.astSeg[u32SegCnt].u16SrcNum;
            pstComfParam->astCtrl[u32CtrlCnt].u32DstNum = pstComfParam->stModel.astSeg[u32SegCnt].u16DstNum;
            memcpy(&pstComfParam->astCtrl[u32CtrlCnt].stTmpBuf, &pstComfParam->stTmpBuf, sizeof(SVP_MEM_INFO_S));
            memcpy(&pstComfParam->astCtrl[u32CtrlCnt].stTskBuf, &pstComfParam->astTskBuf[0], sizeof(SVP_MEM_INFO_S));
            u32CtrlCnt++;
        }
    }
    return s32Ret;

Fail7:
    for (i = 0; i < u32RPNCnt; ++i)
        SvpSampleFreeBlob(&pstComfParam->stRPN[i]);
Fail6:
    for (i = 0; i < u32DstCnt; ++i)
    {
        SvpSampleFreeBlob(&pstComfParam->astDst[i]);
    }
    for (i = 0; i < u32SrcCnt; ++i)
    {
        SvpSampleFreeBlob(&pstComfParam->astSrc[i]);
    }
Fail5:
    for (i = 0; i < u16SrcNum; ++i)
    {
        if (!pstComfParam->fpSrc[i])
        {
            fclose(pstComfParam->fpSrc[i]);
            pstComfParam->fpSrc[i] = NULL;
        }
    }
Fail4:
    SvpSampleMemFree(&pstComfParam->astTskBuf[0]);
Fail3:
    SvpSampleMemFree(&pstComfParam->stTmpBuf);
Fail2:
    HI_MPI_SVP_NNIE_UnloadModel(&(pstComfParam->stModel));
Fail1:
    SvpSampleMemFree(&pstComfParam->stModelBuf);

    return s32Ret;
}


//多阶段卷积网络申请的内存注销
void SvpSampleMultiSegCnnDeinit(SVP_NNIE_MULTI_SEG_S *pstComParam)
{
    HI_U32 i, j;
    HI_U32 u32DstCnt = 0, u32SrcCnt = 0, u32RPNCnt = 0;
    if (!pstComParam)
    {
        printf("pstComParma is NULL\n");
        return;
    }

    for (i = 0; i < pstComParam->stModel.u32NetSegNum; ++i)//段数量
    {
        for (j = 0; j < pstComParam->stModel.astSeg[i].u16DstNum; ++j)//输出数量
        {
            SvpSampleFreeBlob(&pstComParam->astDst[j + u32DstCnt]);
        }
        u32DstCnt += pstComParam->stModel.astSeg[i].u16DstNum;

        for (j = 0; j < pstComParam->stModel.astSeg[i].u16SrcNum; ++j)//输入数量
        {
            SvpSampleFreeBlob(&pstComParam->astSrc[j + u32SrcCnt]);
            SvpSampleCloseFile(pstComParam->fpSrc[j + u32SrcCnt]);
        }
        u32SrcCnt += pstComParam->stModel.astSeg[i].u16SrcNum;

        for (j = 0; j < pstComParam->stModel.astSeg[i].u16RoiPoolNum; ++j)//RPN数量
        {
            SvpSampleFreeBlob(&pstComParam->stRPN[j + u32RPNCnt]);
        }
        u32RPNCnt += pstComParam->stModel.astSeg[i].u16RoiPoolNum;
    }

    SvpSampleMemFree(&pstComParam->astTskBuf[0]);
    SvpSampleMemFree(&pstComParam->stTmpBuf);
    HI_MPI_SVP_NNIE_UnloadModel(&(pstComParam->stModel));
    SvpSampleMemFree(&pstComParam->stModelBuf);

    memset(pstComParam, 0, sizeof(SVP_NNIE_MULTI_SEG_S));
}


//打印检测结果，参数：检测结果的结构体，检测框个数，结果文件路径(文件夹名，文件名的前缀)
void SvpSampleDetectionPrint(const SVP_SAMPLE_BOX_RESULT_INFO_S *pstResultBoxesInfo,HI_U32 u32BoxNum,
    string& strResultFolderDir, SVP_SAMPLE_FILE_NAME_PAIR& imgNamePair)
{
    if ((NULL == pstResultBoxesInfo) || (pstResultBoxesInfo->pstBbox == NULL)) return;

    HI_U32 i = 0;

    /* e.g. result_SVP_SAMPLE_YOLO_V1/dog_bike_car_448x448_detResult.txt */
    string fileName = strResultFolderDir + imgNamePair.first + "_detResult.txt";
    ofstream fout(fileName.c_str());
    if (!fout.good()) {
        printf("%s open failure!", fileName.c_str());
        return;
    }

    PrintBreakLine(HI_TRUE);

    /* detResult start with origin image width and height */
    fout << pstResultBoxesInfo->u32OriImWidth << "  " << pstResultBoxesInfo->u32OriImHeight << endl;
    cout << pstResultBoxesInfo->u32OriImWidth << "  " << pstResultBoxesInfo->u32OriImHeight << endl;

    //printf("imgName\tclass\tconfidence\txmin\tymin\txmax\tymax\n");

    for(i = 0;i < u32BoxNum;i++)
    {
        HI_CHAR resultLine[512];

        snprintf(resultLine, 512, "%s  %4d  %9.8f  %4.2f  %4.2f  %4.2f  %4.2f\n",
            imgNamePair.first.c_str(),
            pstResultBoxesInfo->pstBbox[i].u32MaxScoreIndex,
            pstResultBoxesInfo->pstBbox[i].f32ClsScore,
            pstResultBoxesInfo->pstBbox[i].f32Xmin, pstResultBoxesInfo->pstBbox[i].f32Ymin,
            pstResultBoxesInfo->pstBbox[i].f32Xmax, pstResultBoxesInfo->pstBbox[i].f32Ymax);

        fout << resultLine;
        cout << resultLine;
    }

    PrintBreakLine(HI_TRUE);

    fout.close();
}
