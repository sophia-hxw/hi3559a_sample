#include "SvpSampleWk.h"
#include "SvpSampleCom.h"
#include "mpi_nnie.h"
#ifdef USE_OPENCV
#include "cv_write_segment.h"
#endif

const HI_CHAR *paszPicList_segnet[][SVP_NNIE_MAX_INPUT_NUM] =
{
    { "../data/segmentation/segnet/image_test_list.txt" }
};

#ifndef USE_FUNC_SIM /* inst wk */
const HI_CHAR *pszModelName_segnet[] = { "../data/segmentation/segnet/inst/segnet_inst.wk","" };
#else /* func wk */
const HI_CHAR *pszModelName_segnet[] = { "../data/segmentation/segnet/inst/segnet_func.wk","" };
#endif

static HI_S32 SvpSampleSegnetForword(SVP_NNIE_ONE_SEG_S *pstClfParam, SVP_NNIE_CFG_S *pstClfCfg)
{
    HI_S32 s32Ret = HI_SUCCESS;
    SVP_NNIE_HANDLE SvpNnieHandle;
    SVP_NNIE_ID_E enNnieId = SVP_NNIE_ID_0;
    HI_BOOL bInstant = HI_TRUE;
    HI_BOOL bFinish = HI_FALSE;
    HI_BOOL bBlock = HI_TRUE;

    s32Ret = HI_MPI_SVP_NNIE_Forward(&SvpNnieHandle, pstClfParam->astSrc, &pstClfParam->stModel,
        pstClfParam->astDst, &pstClfParam->stCtrl, bInstant);
    CHECK_EXP_RET(HI_SUCCESS != s32Ret, s32Ret, "Error(%#x): CNN_Forward failed!", s32Ret);

    s32Ret = HI_MPI_SVP_NNIE_Query(enNnieId, SvpNnieHandle, &bFinish, bBlock);
    while (HI_ERR_SVP_NNIE_QUERY_TIMEOUT == s32Ret)
    {
        USLEEP(100);
        s32Ret = HI_MPI_SVP_NNIE_Query(enNnieId, SvpNnieHandle, &bFinish, bBlock);
    }
    CHECK_EXP_RET(HI_SUCCESS != s32Ret, s32Ret, "Error(%#x): query failed!", s32Ret);

    return HI_SUCCESS;
}

/*classification with input images and labels, print the top-N result */
HI_S32 SvpSampleSegnet(const HI_CHAR *pszModelName, const HI_CHAR *paszPicList[], HI_S32 s32Cnt)
{
    HI_S32 s32Ret = HI_SUCCESS;

    HI_U32 i, j;
    HI_U32 u32TopN = 5;
    HI_U32 u32MaxInputNum = 10;
    HI_U32 u32Batch, u32LoopCnt, u32StartId;

    SVP_NNIE_ONE_SEG_S stClfParam = { 0 };
    SVP_NNIE_CFG_S     stClfCfg = { 0 };

    /* mkdir to save result, name folder by model type */
    string strNetType = "SVP_SAMPLE_SEGNET";
    string strResultFolderDir = "result_" + strNetType + "/";
    _mkdir(strResultFolderDir.c_str());

    vector<SVP_SAMPLE_FILE_NAME_PAIR> imgNameRecoder;

    stClfCfg.pszModelName = pszModelName;
    memcpy(&stClfCfg.paszPicList, paszPicList, sizeof(HI_VOID*)*s32Cnt);
    stClfCfg.u32MaxInputNum = u32MaxInputNum;
    stClfCfg.u32TopN = u32TopN;
    stClfCfg.bNeedLabel = HI_FALSE;

    s32Ret = SvpSampleOneSegCnnInit(&stClfCfg, &stClfParam);
    CHECK_EXP_RET(HI_SUCCESS != s32Ret, s32Ret, "SvpSampleOneSegCnnInitMem failed");

    // assure that there is enough mem in one batch
    u32Batch = SVP_SAMPLE_MIN(u32MaxInputNum, stClfParam.u32TotalImgNum);
    u32Batch = SVP_SAMPLE_MIN(u32Batch, stClfParam.astSrc[0].u32Num);
    CHECK_EXP_GOTO(0 == u32Batch, Fail,
        "u32Batch = 0 failed! u32MaxInputNum(%d), tClfParam.u32TotalImgNum(%d), astSrc[0].u32Num(%d)",
        u32MaxInputNum, stClfParam.u32TotalImgNum, stClfParam.astSrc[0].u32Num);

    u32LoopCnt = stClfParam.u32TotalImgNum / u32Batch;

    // process images in batch size of u32Batch
    for (i = 0, u32StartId = 0; i < u32LoopCnt; i++, u32StartId += u32Batch)
    {
        imgNameRecoder.clear();

        s32Ret = SvpSampleReadAllSrcImg(stClfParam.fpSrc, stClfParam.astSrc, stClfParam.stModel.astSeg[0].u16SrcNum, imgNameRecoder);
        CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail, "Error(%#x):SvpSampleReadAllSrcImg failed!", s32Ret);

        CHECK_EXP_GOTO(imgNameRecoder.size() != u32Batch, Fail,
            "Error(%#x):imgNameRecoder.size(%d) != u32Batch(%d)", HI_FAILURE, (HI_U32)imgNameRecoder.size(), u32Batch);

        s32Ret = SvpSampleSegnetForword(&stClfParam, &stClfCfg);
        CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail, "Error(%#x):SvpSampleSegnetForword failed!", s32Ret);

#ifdef USE_OPENCV
        s32Ret = SVPUtils_WriteSegment(&stClfParam.astDst[0], "Snapshot_SEGNET_result.png");
        CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail, "Error(%#x):SVPUtils_WriteSegment failed!", s32Ret);
#endif
    }

    // the rest of images
    u32Batch = stClfParam.u32TotalImgNum - u32StartId;
    if (u32Batch > 0)
    {
        for (j = 0; j < stClfParam.stModel.astSeg[0].u16SrcNum; j++) {
            stClfParam.astSrc[j].u32Num = u32Batch;
        }
        for (j = 0; j < stClfParam.stModel.astSeg[0].u16DstNum; j++) {
            stClfParam.astDst[j].u32Num = u32Batch;
        }

        imgNameRecoder.clear();

        s32Ret = SvpSampleReadAllSrcImg(stClfParam.fpSrc, stClfParam.astSrc, stClfParam.stModel.astSeg[0].u16SrcNum, imgNameRecoder);
        CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail, "Error(%#x):SvpSampleReadAllSrcImg failed!", s32Ret);

        CHECK_EXP_GOTO(imgNameRecoder.size() != u32Batch, Fail,
            "Error(%#x):imgNameRecoder.size(%d) != u32Batch(%d)", HI_FAILURE, (HI_U32)imgNameRecoder.size(), u32Batch);

        s32Ret = SvpSampleSegnetForword(&stClfParam, &stClfCfg);
        CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail, "SvpSampleSegnetForword failed");

#ifdef USE_OPENCV
        s32Ret = SVPUtils_WriteSegment(&stClfParam.astDst[0], "Snapshot_SEGNET_result.png");
        CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail, "Error(%#x):SVPUtils_WriteSegment failed!", s32Ret);
#endif
    }

Fail:
    SvpSampleOneSegCnnDeinit(&stClfParam);

    return HI_SUCCESS;
}

void SvpSampleCnnFcnSegnet()
{
    printf("%s start ...\n", __FUNCTION__);
    SvpSampleSegnet(pszModelName_segnet[0],
        paszPicList_segnet[0]);
    printf("%s end ...\n\n", __FUNCTION__);
    fflush(stdout);
}

