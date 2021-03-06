#include <fstream>

#include "SvpSampleWk.h"
#include "SvpSampleCom.h"

#include "mpi_nnie.h"

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

    printf("ReadWk(%s)\n", pszModelName);

    // 文件的结尾
    fseek(pfModel, 0, SEEK_END);
    // 长度
    pstModelBuf->u32Size = ftell(pfModel);
    // 文件的开头
    fseek(pfModel, 0, SEEK_SET);

    s32Ret = SvpSampleMallocMem(NULL, NULL, pstModelBuf->u32Size, pstModelBuf);
    CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail, "Error(%#x): Malloc model buf failed!", s32Ret);

    // 从给定流 pfModel 读取数据到 pstModelBuf->u64VirAddr 所指向的数组中
    u32Cnt = (HI_U32)fread((void*)pstModelBuf->u64VirAddr, pstModelBuf->u32Size, 1, pfModel);
    if (1 != u32Cnt)
    {
        s32Ret = HI_FAILURE;
    }

Fail:
    SvpSampleCloseFile(pfModel);

    return s32Ret;
}

template<class sT>
static HI_S32 SvpSampleLoadImageList(HI_CHAR aszImg[SVP_SAMPLE_MAX_PATH], const SVP_NNIE_CFG_S *pstClfCfg, sT *pstComfParam)
{
    HI_U16 u16SrcNum = pstComfParam->stModel.astSeg[0].u16SrcNum;
    HI_U32 u32Num = 0;

    CHECK_EXP_RET(pstClfCfg->paszPicList[0] == NULL, HI_ERR_SVP_NNIE_ILLEGAL_PARAM,
        "Error(%#x): input pic_list[0] is null", HI_ERR_SVP_NNIE_ILLEGAL_PARAM);

    pstComfParam->fpSrc[0] = SvpSampleOpenFile(pstClfCfg->paszPicList[0], "r");
    CHECK_EXP_RET(pstComfParam->fpSrc[0] == NULL, HI_ERR_SVP_NNIE_OPEN_FILE,
        "Error(%#x), Open file(%s) failed!", HI_ERR_SVP_NNIE_OPEN_FILE, pstClfCfg->paszPicList[0]);

    //从指定的流 pstComfParam->fpSrc[0] 读取一行，并把它存储在 aszImg 所指向的字符串内
    while (fgets(aszImg, SVP_SAMPLE_MAX_PATH, pstComfParam->fpSrc[0]) != NULL)
    {
        u32Num++;
    }
    pstComfParam->u32TotalImgNum = u32Num;

    for (HI_U32 i = 1; i < u16SrcNum; i++) {
        u32Num = 0;
        CHECK_EXP_GOTO(pstClfCfg->paszPicList[i] == NULL, FAIL,
            "u16SrcNum = %d, but the %dth input pic_list file is null", u16SrcNum, i);

        pstComfParam->fpSrc[i] = SvpSampleOpenFile(pstClfCfg->paszPicList[i], "r");
        CHECK_EXP_GOTO(pstComfParam->fpSrc[i] == NULL, FAIL, "Error(%#x), Open file(%s) failed!",
            HI_ERR_SVP_NNIE_OPEN_FILE, pstClfCfg->paszPicList[i]);

        while (fgets(aszImg, SVP_SAMPLE_MAX_PATH, pstComfParam->fpSrc[i]) != NULL) {
            u32Num++;
        }
        CHECK_EXP_GOTO(u32Num != pstComfParam->u32TotalImgNum, FAIL,
            "The %dth pic_list file has a num of %d, which is not equal to %d",
            i, u32Num, pstComfParam->u32TotalImgNum);
    }

    return HI_SUCCESS;

FAIL:
    for (HI_U16 i = 0; i < u16SrcNum; ++i) {
        SvpSampleCloseFile(pstComfParam->fpSrc[i]);
    }
    return HI_FAILURE;
}

static HI_S32 SvpSampleOpenLabelList(HI_CHAR aszImg[SVP_SAMPLE_MAX_PATH], const SVP_NNIE_CFG_S *pstClfCfg, SVP_NNIE_ONE_SEG_S *pstComfParam)
{
    HI_U16 u16DstNum = pstComfParam->stModel.astSeg[0].u16DstNum;

    if (pstClfCfg->bNeedLabel)
    {
        // all input label file should have the same num of labels of input image
        for (HI_U32 i = 0; i < u16DstNum; i++)
        {
            HI_U32 u32Num = 0;
            CHECK_EXP_RET(!pstClfCfg->paszLabel[i], HI_ERR_SVP_NNIE_ILLEGAL_PARAM,
                "u16DstNum = %d, but the %dth input label file is null", u16DstNum, i);

            pstComfParam->fpLabel[i] = SvpSampleOpenFile(pstClfCfg->paszLabel[i], "r");
            CHECK_EXP_GOTO(!(pstComfParam->fpLabel[i]), FAIL, "Error(%#x), Open file(%s) failed!",
                HI_ERR_SVP_NNIE_OPEN_FILE, pstClfCfg->paszLabel[i]);
            while (fgets(aszImg, SVP_SAMPLE_MAX_PATH, pstComfParam->fpLabel[i]) != NULL)
            {
                u32Num++;
            }

            CHECK_EXP_GOTO(u32Num != pstComfParam->u32TotalImgNum, FAIL,
                "The %dth label file has a num of %d, which is not equal to %d",
                i, u32Num, pstComfParam->u32TotalImgNum);
        }

    }

    return HI_SUCCESS;

FAIL:
    for (HI_U16 i = 0; i < u16DstNum; ++i) {
        SvpSampleCloseFile(pstComfParam->fpLabel[i]);
    }
    return HI_FAILURE;
}

static HI_S32 SvpSampleAllocBlobMemClf(const SVP_NNIE_CFG_S *pstClfCfg, SVP_NNIE_ONE_SEG_S *pstComfParam, SVP_SAMPLE_LSTMRunTimeCtx *pstLSTMCtx)
{
    HI_S32 s32Ret = HI_SUCCESS;

    SVP_NNIE_MODEL_S* pstModel = &pstComfParam->stModel;
    SVP_NNIE_SEG_S* astSeg = pstModel->astSeg;
    HI_U16 u16SrcNum = astSeg[0].u16SrcNum;
    HI_U16 u16DstNum = astSeg[0].u16DstNum;
    HI_U32 u32Num = SVP_SAMPLE_MIN(pstComfParam->u32TotalImgNum, pstClfCfg->u32MaxInputNum);
    HI_U32 u32MaxClfNum = 0;

    // malloc src, dst blob buf
    for (HI_U32 u32SegCnt = 0; u32SegCnt < pstModel->u32NetSegNum; ++u32SegCnt)
    {
        SVP_NNIE_NODE_S* pstSrcNode = (SVP_NNIE_NODE_S*)(astSeg[u32SegCnt].astSrcNode);
        SVP_NNIE_NODE_S* pstDstNode = (SVP_NNIE_NODE_S*)(astSeg[u32SegCnt].astDstNode);

        // malloc src blob buf;
        for (HI_U16 i = 0; i < astSeg[u32SegCnt].u16SrcNum; ++i)
        {
            SVP_BLOB_TYPE_E enType = pstSrcNode->enType;
            if (SVP_BLOB_TYPE_SEQ_S32 == enType)
            {
                HI_U32 u32Dim = pstSrcNode->unShape.u32Dim;
                s32Ret = SvpSampleMallocSeqBlob(&pstComfParam->astSrc[i], enType, u32Num, u32Dim, pstLSTMCtx);
            }
            else
            {
                HI_U32 u32SrcC = pstSrcNode->unShape.stWhc.u32Chn;
                HI_U32 u32SrcW = pstSrcNode->unShape.stWhc.u32Width;
                HI_U32 u32SrcH = pstSrcNode->unShape.stWhc.u32Height;
                s32Ret = SvpSampleMallocBlob(&pstComfParam->astSrc[i],
                    enType, u32Num, u32SrcC, u32SrcW, u32SrcH);
            }
            CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, FAIL, "Error(%#x): Malloc src blob failed!", s32Ret);
            ++pstSrcNode;
        }

        // malloc dst blob buf;
        for (HI_U16 i = 0; i < astSeg[u32SegCnt].u16DstNum; ++i)
        {
            SVP_BLOB_TYPE_E enType = pstDstNode->enType;
            if (SVP_BLOB_TYPE_SEQ_S32 == enType)
            {
                HI_U32 u32Dim = pstDstNode->unShape.u32Dim;
                s32Ret = SvpSampleMallocSeqBlob(&pstComfParam->astDst[i], enType, u32Num, u32Dim, pstLSTMCtx);
            }
            else
            {
                HI_U32 u32DstC = pstDstNode->unShape.stWhc.u32Chn;
                HI_U32 u32DstW = pstDstNode->unShape.stWhc.u32Width;
                HI_U32 u32DstH = pstDstNode->unShape.stWhc.u32Height;

                s32Ret = SvpSampleMallocBlob(&pstComfParam->astDst[i],
                    enType, u32Num, u32DstC, u32DstW, u32DstH);
            }
            CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, FAIL, "Error(%#x): Malloc dst blob failed!", s32Ret);

            // normal classification net which has FC layer before the last softmax layer
            if (pstComfParam->astDst[i].enType == SVP_BLOB_TYPE_VEC_S32) {
                pstComfParam->au32ClfNum[i] = pstComfParam->astDst[i].unShape.stWhc.u32Width;
            }

            // classification net, such as squeezenet, which has global_pooling layer before the last softmax layer
            else {
                pstComfParam->au32ClfNum[i] = pstComfParam->astDst[i].unShape.stWhc.u32Chn;
            }

            if (u32MaxClfNum < pstComfParam->astDst[i].unShape.stWhc.u32Width) {
                u32MaxClfNum = pstComfParam->au32ClfNum[i];
            }

            ++pstDstNode;
        }

        // memory need by post-process of getting top-N
        if (pstClfCfg->bNeedLabel)
        {
            // memory of single output max dim
            // check u32MaxClfNum > 0
            pstComfParam->pstMaxClfIdScore = (SVP_SAMPLE_CLF_RES_S*)malloc(u32MaxClfNum * sizeof(SVP_SAMPLE_CLF_RES_S));
            CHECK_EXP_GOTO(!pstComfParam->pstMaxClfIdScore, FAIL, "Error: Malloc pstMaxclfIdScore failed!");
            memset(pstComfParam->pstMaxClfIdScore, 0, u32MaxClfNum* sizeof(SVP_SAMPLE_CLF_RES_S));

            for (HI_U16 i = 0; i < astSeg[u32SegCnt].u16DstNum; ++i)
            {
                // memory of TopN with u32Num input
                // check u32Num and u32TopN > 0
                pstComfParam->pastClfRes[i] = (SVP_SAMPLE_CLF_RES_S*)malloc(u32Num * pstClfCfg->u32TopN * sizeof(SVP_SAMPLE_CLF_RES_S));
                CHECK_EXP_GOTO(!pstComfParam->pastClfRes[i], FAIL, "Error: Malloc pastClfRes[%d] failed!", i);
                memset(pstComfParam->pastClfRes[i], 0, u32Num * pstClfCfg->u32TopN * sizeof(SVP_SAMPLE_CLF_RES_S));

            }
        }
        else {
            pstComfParam->fpLabel[0] = NULL;
        }
    }

    return s32Ret;

FAIL:
    SvpSampleMemFree(pstComfParam->pstMaxClfIdScore);
    for (HI_U16 i = 0; i < u16DstNum; ++i) {
        SvpSampleFreeBlob(&pstComfParam->astDst[i]);
        SvpSampleMemFree(pstComfParam->pastClfRes[i]);
    }
    for (HI_U16 i = 0; i < u16SrcNum; ++i) {
        SvpSampleFreeBlob(&pstComfParam->astSrc[i]);
    }

    // some fail goto not mark s32Ret as HI_FAILURE, set it to HI_FAILURE
    // keep s32Ret value if it is not HI_SUCCESS
    if (HI_SUCCESS == s32Ret) {
        s32Ret = HI_FAILURE;
    }

    return s32Ret;
}

static HI_S32 SvpSampleAllocBlobMemDet(
    const HI_U32 *pu32SrcAlign,
    const HI_U32 *pu32DstAlign,
    const SVP_NNIE_CFG_S *pstClfCfg,
    SVP_NNIE_ONE_SEG_DET_S *pstComfParam)
{
    HI_S32 s32Ret = HI_SUCCESS;

    SVP_NNIE_MODEL_S* pstModel = &pstComfParam->stModel;//loadModel出来的模型结构体
    SVP_NNIE_SEG_S* astSeg = pstModel->astSeg;//nnie框架上一个segment的信息

    HI_U16 u16SrcNum = astSeg[0].u16SrcNum;
    HI_U16 u16DstNum = astSeg[0].u16DstNum;
    HI_U32 u32Num = SVP_SAMPLE_MIN(pstComfParam->u32TotalImgNum, pstClfCfg->u32MaxInputNum);

    // malloc src, dst blob buf
    //每个segment中src, dst blob的存储
    for (HI_U32 u32SegCnt = 0; u32SegCnt < pstModel->u32NetSegNum; ++u32SegCnt)
    {
        SVP_NNIE_NODE_S* pstSrcNode = (SVP_NNIE_NODE_S*)(astSeg[u32SegCnt].astSrcNode);
        SVP_NNIE_NODE_S* pstDstNode = (SVP_NNIE_NODE_S*)(astSeg[u32SegCnt].astDstNode);

        // malloc src blob buf;
        for (HI_U16 i = 0; i < pstComfParam->stModel.astSeg[u32SegCnt].u16SrcNum; ++i)
        {
            SVP_BLOB_TYPE_E enType = pstSrcNode->enType;
            HI_U32 u32SrcC = pstSrcNode->unShape.stWhc.u32Chn;
            HI_U32 u32SrcW = pstSrcNode->unShape.stWhc.u32Width;
            HI_U32 u32SrcH = pstSrcNode->unShape.stWhc.u32Height;
            s32Ret = SvpSampleMallocBlob(&pstComfParam->astSrc[i],
                enType, u32Num, u32SrcC, u32SrcW, u32SrcH, pu32SrcAlign ? pu32SrcAlign[i] : STRIDE_ALIGN);
            CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, FAIL, "Error(%#x): Malloc src blob failed!", s32Ret);
            ++pstSrcNode;
        }

        // malloc dst blob buf;
        for (HI_U16 i = 0; i < astSeg[u32SegCnt].u16DstNum; ++i)
        {
            SVP_BLOB_TYPE_E enType = pstDstNode->enType;
            HI_U32 u32DstC = pstDstNode->unShape.stWhc.u32Chn;
            HI_U32 u32DstW = pstDstNode->unShape.stWhc.u32Width;
            HI_U32 u32DstH = pstDstNode->unShape.stWhc.u32Height;

            s32Ret = SvpSampleMallocBlob(&pstComfParam->astDst[i],
                enType, u32Num, u32DstC, u32DstW, u32DstH, pu32DstAlign ? pu32DstAlign[i] : STRIDE_ALIGN);
            CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, FAIL, "Error(%#x): Malloc dst blob failed!", s32Ret);
            ++pstDstNode;
        }
    }

    return s32Ret;
FAIL:
    for (HI_U16 i = 0; i < u16SrcNum; ++i) {
        SvpSampleFreeBlob(&pstComfParam->astSrc[i]);
    }
    for (HI_U16 i = 0; i < u16DstNum; ++i) {
        SvpSampleFreeBlob(&pstComfParam->astDst[i]);
    }

    // some fail goto not mark s32Ret as HI_FAILURE, set it to HI_FAILURE
    // keep s32Ret value if it is not HI_SUCCESS
    if (HI_SUCCESS == s32Ret) {
        s32Ret = HI_FAILURE;
    }

    return s32Ret;
}

static HI_S32 SvpSampleAllocBlobMemMultiSeg(
    const HI_U32 *pu32SrcAlign,
    const HI_U32 *pu32DstAlign,
    const SVP_NNIE_CFG_S *pstComCfg,
    SVP_NNIE_MULTI_SEG_S *pstComfParam)
{
    HI_S32 s32Ret = HI_SUCCESS;

    SVP_NNIE_MODEL_S* pstModel = &pstComfParam->stModel;
    SVP_NNIE_SEG_S* astSeg = pstModel->astSeg;

    HI_U32 u32Num = SVP_SAMPLE_MIN(pstComfParam->u32TotalImgNum, pstComCfg->u32MaxInputNum);
    HI_U32 u32DstCnt = 0, u32SrcCnt = 0, u32RPNCnt = 0;

    // malloc src, dst blob buf
    for (HI_U32 u32SegCnt = 0; u32SegCnt <pstModel->u32NetSegNum; ++u32SegCnt)
    {
        SVP_NNIE_NODE_S* pstSrcNode = (SVP_NNIE_NODE_S*)(astSeg[u32SegCnt].astSrcNode);
        SVP_NNIE_NODE_S* pstDstNode = (SVP_NNIE_NODE_S*)(astSeg[u32SegCnt].astDstNode);

        // malloc src blob buf;
        for (HI_U16 i = 0; i < pstComfParam->stModel.astSeg[u32SegCnt].u16SrcNum; ++i)
        {
            SVP_BLOB_TYPE_E enType = pstSrcNode->enType;
            HI_U32 u32SrcC = pstSrcNode->unShape.stWhc.u32Chn;
            HI_U32 u32SrcW = pstSrcNode->unShape.stWhc.u32Width;
            HI_U32 u32SrcH = pstSrcNode->unShape.stWhc.u32Height;
            s32Ret = SvpSampleMallocBlob(&pstComfParam->astSrc[i + u32SrcCnt],
                enType, u32Num, u32SrcC, u32SrcW, u32SrcH, pu32SrcAlign ? pu32SrcAlign[i] : STRIDE_ALIGN);
            CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, FAIL, "Error(%#x): Malloc src blob failed!", s32Ret);

            ++pstSrcNode;
        }

        u32SrcCnt += astSeg[u32SegCnt].u16SrcNum;

        // malloc dst blob buf;
        for (HI_U16 i = 0; i < astSeg[u32SegCnt].u16DstNum; ++i)
        {
            SVP_BLOB_TYPE_E enType = pstDstNode->enType;
            HI_U32 u32DstC = pstDstNode->unShape.stWhc.u32Chn;
            HI_U32 u32DstW = pstDstNode->unShape.stWhc.u32Width;
            HI_U32 u32DstH = pstDstNode->unShape.stWhc.u32Height;

            HI_U32 u32NumWithBbox = (pstComCfg->u32MaxBboxNum > 0 && astSeg[u32SegCnt].u16RoiPoolNum > 0) ?
                u32Num*pstComCfg->u32MaxBboxNum : u32Num;
            s32Ret = SvpSampleMallocBlob(&pstComfParam->astDst[i + u32DstCnt],
                enType, u32NumWithBbox, u32DstC, u32DstW, u32DstH, pu32DstAlign ? pu32DstAlign[i] : STRIDE_ALIGN);
            CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, FAIL, "Error(%#x): Malloc dst blob failed!", s32Ret);

            ++pstDstNode;
        }
        u32DstCnt += astSeg[u32SegCnt].u16DstNum;

        //malloc RPN blob buf if exists
        if (pstComCfg->u32MaxBboxNum > 0)
        {
            for (HI_U16 i = 0; i < astSeg[u32SegCnt].u16RoiPoolNum; ++i)
            {
                s32Ret = SvpSampleMallocRPNBlob(&pstComfParam->stRPN[u32RPNCnt + i], pstComCfg->u32MaxBboxNum, STRIDE_ALIGN);
                CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, FAIL, "Error(%#x): Malloc rpn blob failed!", s32Ret)
            }
            u32RPNCnt += astSeg[u32SegCnt].u16RoiPoolNum;
        }
    }

    return s32Ret;

FAIL:
    for (HI_U32 i = 0; i < u32RPNCnt; ++i) {
        SvpSampleFreeBlob(&pstComfParam->stRPN[i]);
    }
    for (HI_U32 i = 0; i < u32DstCnt; ++i) {
        SvpSampleFreeBlob(&pstComfParam->astDst[i]);
    }
    for (HI_U32 i = 0; i < u32SrcCnt; ++i) {
        SvpSampleFreeBlob(&pstComfParam->astSrc[i]);
    }

    // some fail goto not mark s32Ret as HI_FAILURE, set it to HI_FAILURE
    // keep s32Ret value if it is not HI_SUCCESS
    if (HI_SUCCESS == s32Ret) {
        s32Ret = HI_FAILURE;
    }

    return s32Ret;
}

template<class sT>
static HI_S32 SvpSampleSetCtrlParamOneSeg(sT *pstComfParam)
{
    SVP_NNIE_FORWARD_CTRL_S* pstCtrl = &pstComfParam->stCtrl;
    SVP_NNIE_SEG_S* astSeg = pstComfParam->stModel.astSeg;

    pstCtrl->enNnieId = SVP_NNIE_ID_0;
    pstCtrl->u32NetSegId = 0;
    pstCtrl->u32SrcNum = astSeg[0].u16SrcNum;
    pstCtrl->u32DstNum = astSeg[0].u16DstNum;

    memcpy(&pstCtrl->stTmpBuf, &pstComfParam->stTmpBuf, sizeof(SVP_MEM_INFO_S));
    memcpy(&pstCtrl->stTskBuf, &pstComfParam->stTskBuf, sizeof(SVP_MEM_INFO_S));

    return HI_SUCCESS;
}

static HI_S32 SvpSampleSetCtrlParamMultiSeg(SVP_NNIE_MULTI_SEG_S *pstComfParam)
{
    SVP_NNIE_FORWARD_CTRL_S* astCtrl = pstComfParam->astCtrl;
    SVP_NNIE_FORWARD_WITHBBOX_CTRL_S* astBboxCtrl = pstComfParam->astBboxCtrl;
    SVP_NNIE_SEG_S* astSeg = pstComfParam->stModel.astSeg;

    HI_U32 u32CtrlCnt = 0, u32BboxCtrlCnt = 0;

    for (HI_U32 u32SegCnt = 0; u32SegCnt < pstComfParam->stModel.u32NetSegNum; ++u32SegCnt)
    {
        if (SVP_NNIE_NET_TYPE_ROI == astSeg[u32SegCnt].enNetType)
        {
            astBboxCtrl[u32BboxCtrlCnt].enNnieId = SVP_NNIE_ID_0;
            astBboxCtrl[u32BboxCtrlCnt].u32NetSegId = u32SegCnt;
            astBboxCtrl[u32BboxCtrlCnt].u32ProposalNum = 1;
            astBboxCtrl[u32BboxCtrlCnt].u32SrcNum = astSeg[u32SegCnt].u16SrcNum;
            astBboxCtrl[u32BboxCtrlCnt].u32DstNum = astSeg[u32SegCnt].u16DstNum;
            memcpy(&astBboxCtrl[u32BboxCtrlCnt].stTmpBuf, &pstComfParam->stTmpBuf, sizeof(SVP_MEM_INFO_S));
            memcpy(&astBboxCtrl[u32BboxCtrlCnt].stTskBuf, &pstComfParam->astTskBuf[0], sizeof(SVP_MEM_INFO_S));
            u32BboxCtrlCnt++;
        }
        else
        {
            astCtrl[u32CtrlCnt].enNnieId = SVP_NNIE_ID_0;
            astCtrl[u32CtrlCnt].u32NetSegId = u32SegCnt;
            astCtrl[u32CtrlCnt].u32SrcNum = astSeg[u32SegCnt].u16SrcNum;
            astCtrl[u32CtrlCnt].u32DstNum = astSeg[u32SegCnt].u16DstNum;
            memcpy(&astCtrl[u32CtrlCnt].stTmpBuf, &pstComfParam->stTmpBuf, sizeof(SVP_MEM_INFO_S));
            memcpy(&astCtrl[u32CtrlCnt].stTskBuf, &pstComfParam->astTskBuf[0], sizeof(SVP_MEM_INFO_S));
            u32CtrlCnt++;
        }
    }

    return HI_SUCCESS;
}

static HI_S32 SvpSampleOneSegCommonInit(SVP_NNIE_CFG_S *pstClfCfg, SVP_NNIE_ONE_SEG_S *pstComfParam, SVP_SAMPLE_LSTMRunTimeCtx *pstLSTMCtx)
{
    HI_S32 s32Ret = HI_SUCCESS;
    HI_CHAR aszImg[SVP_SAMPLE_MAX_PATH] = { '\0' };

    HI_U16 u16SrcNum = 0;
    HI_U16 u16DstNum = 0;

    SVP_MEM_INFO_S *pstModelBuf = &pstComfParam->stModelBuf;
    SVP_MEM_INFO_S *pstTmpBuf   = &pstComfParam->stTmpBuf;
    SVP_MEM_INFO_S *pstTskBuf   = &pstComfParam->stTskBuf;

    /******************** step1, load wk file, *******************************/
    s32Ret = SvpSampleReadWK(pstClfCfg->pszModelName, pstModelBuf);
    CHECK_EXP_RET(HI_SUCCESS != s32Ret, s32Ret, "Error(%#x): read model file(%s) failed", s32Ret, pstClfCfg->pszModelName);

    s32Ret = HI_MPI_SVP_NNIE_LoadModel(pstModelBuf, &(pstComfParam->stModel));
    CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail1, "Error(%#x): LoadModel from %s failed!", s32Ret, pstClfCfg->pszModelName);

    u16SrcNum = pstComfParam->stModel.astSeg[0].u16SrcNum;
    u16DstNum = pstComfParam->stModel.astSeg[0].u16DstNum;

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
    s32Ret = SvpSampleLoadImageList(aszImg, pstClfCfg, pstComfParam);
    CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail4, "Error(%#x): SvpSampleLoadImageList failed!", s32Ret);

    /*********** step6, if need label then open all label file ******************/
    s32Ret = SvpSampleOpenLabelList(aszImg, pstClfCfg, pstComfParam);
    CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail5, "Error(%#x): SvpSampleOpenLabelList failed!", s32Ret);

    /*********** step7, malloc memory of src blob, dst blob and post-process mem ***********/
    s32Ret = SvpSampleAllocBlobMemClf(pstClfCfg, pstComfParam, pstLSTMCtx);
    CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail6, "Error(%#x): SvpSampleAllocBlobMemClf failed!", s32Ret);

    /************************** step8, set ctrl param **************************/
    s32Ret = SvpSampleSetCtrlParamOneSeg(pstComfParam);
    CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail7, "Error(%#x): SvpSampleSetCtrlParamOneSeg failed!", s32Ret);

    return s32Ret;

Fail7:
    SvpSampleMemFree(pstComfParam->pstMaxClfIdScore);

    for (HI_U16 i = 0; i < u16DstNum; ++i) {
        SvpSampleFreeBlob(&pstComfParam->astDst[i]);
        SvpSampleMemFree(pstComfParam->pastClfRes[i]);
    }
    for (HI_U16 i = 0; i < u16SrcNum; ++i) {
        SvpSampleFreeBlob(&pstComfParam->astSrc[i]);
    }

Fail6:
    for (HI_U16 i = 0; i < u16SrcNum; ++i) {
        SvpSampleCloseFile(pstComfParam->fpSrc[i]);
    }

Fail5:
    for (HI_U16 i = 0; i < u16DstNum; ++i) {
        SvpSampleCloseFile(pstComfParam->fpLabel[i]);
    }

Fail4:
    SvpSampleMemFree(&pstComfParam->stTskBuf);
Fail3:
    SvpSampleMemFree(&pstComfParam->stTmpBuf);
Fail2:
    HI_MPI_SVP_NNIE_UnloadModel(&(pstComfParam->stModel));
Fail1:
    SvpSampleMemFree(&pstComfParam->stModelBuf);

    // some fail goto not mark s32Ret as HI_FAILURE, set it to HI_FAILURE
    // keep s32Ret value if it is not HI_SUCCESS
    if (HI_SUCCESS == s32Ret) {
        s32Ret = HI_FAILURE;
    }

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
    if (!pstComParam) {
        printf("pstComParma is NULL\n");
        return;
    }

    for (HI_U32 i = 0; i < pstComParam->stModel.u32NetSegNum; ++i) {
        for (HI_U32 j = 0; j < pstComParam->stModel.astSeg[i].u16DstNum; ++j) {
            SvpSampleFreeBlob(&pstComParam->astDst[j]);
            SvpSampleMemFree(pstComParam->pastClfRes[j]);
        }

        for (HI_U32 j = 0; j < pstComParam->stModel.astSeg[i].u16SrcNum; ++j) {
            SvpSampleFreeBlob(&pstComParam->astSrc[j]);
            SvpSampleCloseFile(pstComParam->fpSrc[j]);
            SvpSampleCloseFile(pstComParam->fpLabel[j]);
        }
    }

    SvpSampleMemFree(pstComParam->pstMaxClfIdScore);
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

//一阶段检测模型初始化，参数：nnie的config参数，一阶段检测网络参数，网络类型
HI_S32 SvpSampleOneSegDetCnnInit(SVP_NNIE_CFG_S *pstClfCfg,
		SVP_NNIE_ONE_SEG_DET_S *pstComfParam, const HI_U8 netType)
{
    HI_S32 s32Ret = HI_SUCCESS;

    // SVP_SAMPLE_MAX_PATH 最长路径名 长度 默认256
    HI_CHAR aszImg[SVP_SAMPLE_MAX_PATH] = { '\0' };
    // 输入节点个数初始化
    HI_U16 u16SrcNum = 0;
    //输出节点个数初始化
    HI_U16 u16DstNum = 0;

    SVP_MEM_INFO_S *pstModelBuf = &pstComfParam->stModelBuf;
    SVP_MEM_INFO_S *pstTmpBuf   = &pstComfParam->stTmpBuf;
    SVP_MEM_INFO_S *pstTskBuf   = &pstComfParam->stTskBuf;

    /******************** step1, load wk file, *******************************/
    s32Ret = SvpSampleReadWK(pstClfCfg->pszModelName, pstModelBuf);
    CHECK_EXP_RET(HI_SUCCESS != s32Ret, s32Ret, "Error(%#x): read model file(%s) failed", s32Ret, pstClfCfg->pszModelName);

    s32Ret = HI_MPI_SVP_NNIE_LoadModel(pstModelBuf, &(pstComfParam->stModel));//模型信息都有了
    CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail1, "Error(%#x): LoadModel from %s failed!", s32Ret, pstClfCfg->pszModelName);

    u16SrcNum = pstComfParam->stModel.astSeg[0].u16SrcNum;// 输入节点个数实际值，从模型处获取
    u16DstNum = pstComfParam->stModel.astSeg[0].u16DstNum;//输出节点个数实际值，从模型处获取

    /******************** step2, malloc tmp_buf *******************************/
    pstComfParam->u32TmpBufSize = pstComfParam->stModel.u32TmpBufSize;
    //TmpBufSize大小，模型才知道

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
    s32Ret = SvpSampleLoadImageList(aszImg, pstClfCfg, pstComfParam);
    CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail4, "Error(%#x): SvpSampleLoadImageList failed!", s32Ret);

    /*********** step6, malloc memory of src blob, dst blob and post-process mem ***********/
    s32Ret = SvpSampleAllocBlobMemDet(NULL, NULL,  pstClfCfg, pstComfParam);
    CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail5, "Error(%#x): SvpSampleAllocBlobMemDet failed!", s32Ret);

    /************************** step7, set ctrl param **************************/
    s32Ret = SvpSampleSetCtrlParamOneSeg(pstComfParam);
    CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail6, "Error(%#x): SvpSampleSetCtrlParamOneSeg failed!", s32Ret);

    return s32Ret;

// 相应步骤有异常，直接条状到相应异常flag点
Fail6:
    for (HI_U16 i = 0; i < u16SrcNum; ++i) {
        SvpSampleFreeBlob(&pstComfParam->astSrc[i]);
    }
    for (HI_U16 i = 0; i < u16DstNum; ++i) {
        SvpSampleFreeBlob(&pstComfParam->astDst[i]);
    }

Fail5:
    for (HI_U16 i = 0; i < u16SrcNum; ++i) {
        SvpSampleCloseFile(pstComfParam->fpSrc[i]);
    }

Fail4:
    SvpSampleMemFree(&pstComfParam->stTskBuf);
Fail3:
    SvpSampleMemFree(&pstComfParam->stTmpBuf);
Fail2:
    HI_MPI_SVP_NNIE_UnloadModel(&(pstComfParam->stModel));
Fail1:
    SvpSampleMemFree(&pstComfParam->stModelBuf);

    // some fail goto not mark s32Ret as HI_FAILURE, set it to HI_FAILURE
    // keep s32Ret value if it is not HI_SUCCESS
    if (HI_SUCCESS == s32Ret) {
        s32Ret = HI_FAILURE;
    }

    return s32Ret;
}

void SvpSampleOneSegDetCnnDeinit(SVP_NNIE_ONE_SEG_DET_S *pstComParam)
{
    if (!pstComParam) {
        printf("pstComParma is NULL\n");
        return;
    }

    SvpSampleMemFree(pstComParam->ps32ResultMem);

    for (HI_U32 i = 0; i < pstComParam->stModel.u32NetSegNum; ++i) {
        for (HI_U32 j = 0; j < pstComParam->stModel.astSeg[i].u16DstNum; ++j) {
            SvpSampleFreeBlob(&pstComParam->astDst[j]);
        }
        for (HI_U32 j = 0; j < pstComParam->stModel.astSeg[i].u16SrcNum; ++j) {
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

HI_S32 SvpSampleMultiSegCnnInit(SVP_NNIE_CFG_S *pstComCfg, SVP_NNIE_MULTI_SEG_S *pstComfParam,
    HI_U32 *pu32SrcAlign, HI_U32 *pu32DstAlign)
{
    HI_S32 s32Ret = HI_SUCCESS;

    HI_CHAR aszImg[SVP_SAMPLE_MAX_PATH] = { '\0' };
    HI_U32 u32MaxTaskSize = 0;
    HI_U32 u32NetSegNum   = 0;
    HI_U16 u16SrcNum      = 0;

    SVP_MEM_INFO_S *pstModelBuf = &pstComfParam->stModelBuf;
    SVP_MEM_INFO_S *pstTmpBuf   = &pstComfParam->stTmpBuf;
    SVP_MEM_INFO_S *pstTskBuf   = &pstComfParam->astTskBuf[0];

    /******************** step1, load wk file, *******************************/
    s32Ret = SvpSampleReadWK(pstComCfg->pszModelName, pstModelBuf);
    CHECK_EXP_RET(HI_SUCCESS != s32Ret, s32Ret, "Error(%#x): read model file(%s) failed", s32Ret, pstComCfg->pszModelName);

    s32Ret = HI_MPI_SVP_NNIE_LoadModel(pstModelBuf, &(pstComfParam->stModel));
    CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail1, "Error(%#x): LoadModel from %s failed!", s32Ret, pstComCfg->pszModelName);

    u32NetSegNum = pstComfParam->stModel.u32NetSegNum;
    u16SrcNum = pstComfParam->stModel.astSeg[0].u16SrcNum;

    /******************** step2, malloc tmp_buf *******************************/
    pstComfParam->u32TmpBufSize = pstComfParam->stModel.u32TmpBufSize;

    s32Ret = SvpSampleMallocMem(NULL, NULL, pstComfParam->u32TmpBufSize, pstTmpBuf);
    CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail2, "Error(%#x): Malloc tmp buf failed!", s32Ret);

    /******************** step3, get task_buf size *******************************/
    CHECK_EXP_GOTO(u32NetSegNum <= 1, Fail3, "netSegNum should be larger than 1");

    s32Ret = HI_MPI_SVP_NNIE_GetTskBufSize(pstComCfg->u32MaxInputNum, pstComCfg->u32MaxBboxNum,
        &pstComfParam->stModel, pstComfParam->au32TaskBufSize, u32NetSegNum);
    CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail3, "Error(%#x): GetTaskSize failed!", s32Ret);

    /******************** step4, malloc tsk_buf size *******************************/
    //NNIE and CPU running at interval. get max task_buf size
    for (HI_U32 i = 0; i < u32NetSegNum; i++) {
        if (u32MaxTaskSize < pstComfParam->au32TaskBufSize[i]) {
            u32MaxTaskSize = pstComfParam->au32TaskBufSize[i];
        }
    }

    s32Ret = SvpSampleMallocMem(NULL, NULL, u32MaxTaskSize, pstTskBuf);
    CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail3, "Error(%#x): Malloc task buf failed!", s32Ret);

    /*********** step5, check and open all input images list file ******************/
    // all input pic_list file should have the same num of input image
    s32Ret = SvpSampleLoadImageList(aszImg, pstComCfg, pstComfParam);
    CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail4, "Error(%#x): SvpSampleLoadImageList failed!", s32Ret);

    /*********** step6, malloc memory of src blob, dst blob and post-process mem ***********/
    s32Ret = SvpSampleAllocBlobMemMultiSeg(pu32SrcAlign, pu32DstAlign, pstComCfg, pstComfParam);
    CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail5, "Error(%#x): SvpSampleAllocBlobMemMultiSeg failed!", s32Ret);

    /************************** step7, set ctrl param **************************/
    s32Ret = SvpSampleSetCtrlParamMultiSeg(pstComfParam);
    CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail6, "Error(%#x): SvpSampleSetCtrlParamMultiSeg failed!", s32Ret);

    return s32Ret;

Fail6:
    for (HI_U32 i = 0; i < SVP_NNIE_MAX_OUTPUT_NUM; ++i) {
        SvpSampleFreeBlob(&pstComfParam->stRPN[i]);
        SvpSampleFreeBlob(&pstComfParam->astDst[i]);
    }
    for (HI_U32 i = 0; i < SVP_NNIE_MAX_INPUT_NUM; ++i) {
        SvpSampleFreeBlob(&pstComfParam->astSrc[i]);
    }

Fail5:
    for (HI_U16 i = 0; i < u16SrcNum; ++i) {
        SvpSampleCloseFile(pstComfParam->fpSrc[i]);
    }

Fail4:
    SvpSampleMemFree(&pstComfParam->astTskBuf[0]);
Fail3:
    SvpSampleMemFree(&pstComfParam->stTmpBuf);
Fail2:
    HI_MPI_SVP_NNIE_UnloadModel(&(pstComfParam->stModel));
Fail1:
    SvpSampleMemFree(&pstComfParam->stModelBuf);

    // some fail goto not mark s32Ret as HI_FAILURE, set it to HI_FAILURE
    // keep s32Ret value if it is not HI_SUCCESS
    if (HI_SUCCESS == s32Ret) {
        s32Ret = HI_FAILURE;
    }

    return s32Ret;
}

void SvpSampleMultiSegCnnDeinit(SVP_NNIE_MULTI_SEG_S *pstComParam)
{
    HI_U32 u32DstCnt = 0, u32SrcCnt = 0, u32RPNCnt = 0;
    if (!pstComParam) {
        printf("pstComParma is NULL\n");
        return;
    }

    for (HI_U32 i = 0; i < pstComParam->stModel.u32NetSegNum; ++i) {
        for (HI_U32 j = 0; j < pstComParam->stModel.astSeg[i].u16DstNum; ++j) {
            SvpSampleFreeBlob(&pstComParam->astDst[j + u32DstCnt]);
        }
        u32DstCnt += pstComParam->stModel.astSeg[i].u16DstNum;

        for (HI_U32 j = 0; j < pstComParam->stModel.astSeg[i].u16SrcNum; ++j) {
            SvpSampleFreeBlob(&pstComParam->astSrc[j + u32SrcCnt]);
            SvpSampleCloseFile(pstComParam->fpSrc[j + u32SrcCnt]);
        }
        u32SrcCnt += pstComParam->stModel.astSeg[i].u16SrcNum;

        for (HI_U32 j = 0; j < pstComParam->stModel.astSeg[i].u16RoiPoolNum; ++j) {
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
