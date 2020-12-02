#include "SvpSampleWk.h"
#include "SvpSampleCom.h"
#include "mpi_nnie.h"
#include "mpi_nnie_ssd.h"

#ifdef USE_OPENCV
#include "cv_draw_rect.h"
#endif

const HI_CHAR *g_paszPicList_d[][SVP_NNIE_MAX_INPUT_NUM] = {
    { "../data/detection/yolov1/image_test_list.txt" },
    { "../data/detection/yolov2/image_test_list.txt" },
    { "../data/detection/ssd/image_test_list.txt" }
};

#ifndef USE_FUNC_SIM /* inst wk */
const HI_CHAR *g_paszModelName_d[] = {
    "../data/detection/yolov1/inst/inst_yolov1_inst.wk",
    "../data/detection/yolov2/inst/inst_yolov2_inst.wk",
    "../data/detection/ssd/inst/inst_ssd_inst.wk"
};
#else /* func wk */
const HI_CHAR *g_paszModelName_d[] = {
    "../data/detection/yolov1/inst/inst_yolov1_func.wk",
    "../data/detection/yolov2/inst/inst_yolov2_func.wk",
    "../data/detection/ssd/inst/inst_ssd_func.wk"
};
#endif

/* the order is same with SVP_SAMPLE_WK_DETECT_NET_FASTER_RCNN_TYPE_E*/
const HI_CHAR *g_paszModelType_d[] = {
    "SVP_SAMPLE_YOLO_V1",
    "SVP_SAMPLE_YOLO_V2",
    "SVP_SAMPLE_SSD",
    "SVP_SAMPLE_DET_UNKNOWN",
};

HI_S32 SvpSampleCnnDetectionForword(SVP_NNIE_ONE_SEG_DET_S *pstDetParam, SVP_NNIE_CFG_S *pstDetCfg)
{
    HI_S32 s32Ret = HI_SUCCESS;
    SVP_NNIE_HANDLE SvpNnieHandle;
    SVP_NNIE_ID_E enNnieId = SVP_NNIE_ID_0;
    HI_BOOL bInstant = HI_TRUE;
    HI_BOOL bFinish  = HI_FALSE;
    HI_BOOL bBlock   = HI_TRUE;

    s32Ret = HI_MPI_SVP_NNIE_Forward(&SvpNnieHandle, pstDetParam->astSrc, &pstDetParam->stModel,
        pstDetParam->astDst, &pstDetParam->stCtrl, bInstant);
    CHECK_EXP_RET(HI_SUCCESS != s32Ret, s32Ret, "Error(%#x): CNN_Forward failed!", s32Ret);
    s32Ret = HI_MPI_SVP_NNIE_Query(enNnieId, SvpNnieHandle, &bFinish, bBlock);
    while (HI_ERR_SVP_NNIE_QUERY_TIMEOUT == s32Ret)
    {
        USLEEP(100);
        s32Ret = HI_MPI_SVP_NNIE_Query(enNnieId, SvpNnieHandle, &bFinish, bBlock);
    }
    CHECK_EXP_RET(HI_SUCCESS != s32Ret, s32Ret, "Error(%#x): query failed!", s32Ret);

    return s32Ret;
}

HI_S32 SvpSampleDetOneSegGetResult(SVP_NNIE_ONE_SEG_DET_S *pstDetParam,
    HI_U8 netType, HI_VOID *pExtraParam,
    string& strResultFolderDir, vector<SVP_SAMPLE_FILE_NAME_PAIR>& imgNameRecoder)
{
    HI_S32 s32Ret = HI_SUCCESS;
    SVP_SAMPLE_BOX_S astBoxesResult[1024];

    HI_U32* p32BoxNum = (HI_U32*)malloc(pstDetParam->astSrc->u32Num * sizeof(HI_U32));
    CHECK_EXP_RET(p32BoxNum == NULL, HI_FAILURE, "malloc p32BoxNum failure!");

    SVP_SAMPLE_BOX_RESULT_INFO_S stBoxesInfo = {0};
    stBoxesInfo.pstBbox = astBoxesResult;
    stBoxesInfo.u32OriImHeight = pstDetParam->astSrc[0].unShape.stWhc.u32Height;
    stBoxesInfo.u32OriImWidth = pstDetParam->astSrc[0].unShape.stWhc.u32Width;

    switch (netType)
    {
    case SVP_SAMPLE_WK_DETECT_NET_YOLOV1:
    {
        SvpSampleWkYoloV1GetResult(pstDetParam->astDst, &stBoxesInfo,
            p32BoxNum, strResultFolderDir, imgNameRecoder);
        break;
    }
    case SVP_SAMPLE_WK_DETECT_NET_YOLOV2:
    {
        SvpSampleWkYoloV2GetResult(pstDetParam->astDst, pstDetParam->ps32ResultMem,
            &stBoxesInfo, p32BoxNum, strResultFolderDir, imgNameRecoder);
        break;
    }
    case SVP_SAMPLE_WK_DETECT_NET_SSD:
    {
        SVP_NNIE_SSD_S *pstSSDParam = (SVP_NNIE_SSD_S *)pExtraParam;
        SvpSampleWkSSDGetResult(pstDetParam, pstSSDParam, &stBoxesInfo,
            p32BoxNum, strResultFolderDir, imgNameRecoder);
        break;
    }
    default:
    {
        printf("the netType is %d, out the range [%d, %d]",netType, SVP_SAMPLE_WK_DETECT_NET_YOLOV1, SVP_SAMPLE_WK_DETECT_NET_SSD);
        s32Ret = HI_FAILURE;
        break;
    }
    }
#ifdef USE_OPENCV
    if (HI_SUCCESS == s32Ret)
    {
        //TBD: batch images result process
        //need get batch astBoxesResult from front detection result calculate
        for (HI_U32 j = 0; j < pstDetParam->astDst->u32Num; j++)
        {
            vector <SVPUtils_TaggedBox_S> vTaggedBoxes;
            for (HI_U32 i = 0; i < p32BoxNum[j]; i++)
            {
                SVPUtils_TaggedBox_S stTaggedBox;
                stTaggedBox.stRect.x = (HI_U32)astBoxesResult[i].f32Xmin;
                stTaggedBox.stRect.y = (HI_U32)astBoxesResult[i].f32Ymin;
                stTaggedBox.stRect.w = (HI_U32)(astBoxesResult[i].f32Xmax - astBoxesResult[i].f32Xmin);
                stTaggedBox.stRect.h = (HI_U32)(astBoxesResult[i].f32Ymax - astBoxesResult[i].f32Ymin);
                stTaggedBox.fScore = astBoxesResult[i].f32ClsScore;
                stTaggedBox.u32Class = astBoxesResult[i].u32MaxScoreIndex;
                vTaggedBoxes.push_back(stTaggedBox);
            }

            string strBoxedImgPath = imgNameRecoder[j].first + "_det.png";
            strBoxedImgPath = strResultFolderDir + strBoxedImgPath;
            SVPUtils_DrawBoxes(pstDetParam->astSrc, RGBPLANAR, strBoxedImgPath.c_str(), vTaggedBoxes, j);
        }
    }
#endif

    free(p32BoxNum);
    return s32Ret;
}

extern void getSSDParm(SVP_NNIE_SSD_S *param, const SVP_NNIE_ONE_SEG_DET_S *pstDetecionParam);

void getSSDResultPara(Result_SSD_Para_S *stParam)
{
    stParam->u32KeepTopK = 200;
    stParam->u32NumClasses = 21;
    stParam->u32TotlBoxNum = 8732;
}

HI_U32 SvpSampleGetResultMemSize(const HI_U8 netType, SVP_NNIE_SSD_S *pstSSDParam)
{
    HI_U32 u32ResultMemSize = 0;
    switch (netType)
    {
    case SVP_SAMPLE_WK_DETECT_NET_YOLOV2:
    {
        HI_U32 inputdate_size = SVP_SAMPLE_YOLOV2_GRIDNUM*SVP_SAMPLE_YOLOV2_GRIDNUM*SVP_SAMPLE_YOLOV2_CHANNLENUM * sizeof(HI_FLOAT);
        HI_U32 u32TmpBoxSize = SVP_SAMPLE_YOLOV2_GRIDNUM*SVP_SAMPLE_YOLOV2_GRIDNUM*SVP_SAMPLE_YOLOV2_CHANNLENUM * sizeof(HI_U32);
        HI_U32 u32BoxSize = SVP_SAMPLE_YOLOV2_BOXTOTLENUM * sizeof(SVP_SAMPLE_BOX_S);
        HI_U32 u32StackSize = SVP_SAMPLE_YOLOV2_BOXTOTLENUM * sizeof(SVP_SAMPLE_STACK_S);
        HI_U32 u32ResultBoxSize = SVP_SAMPLE_YOLOV2_MAX_BOX_NUM * sizeof(SVP_SAMPLE_BOX_S);
        u32ResultMemSize = inputdate_size + u32TmpBoxSize + u32BoxSize + u32StackSize + u32ResultBoxSize;
        break;
    }
    case SVP_SAMPLE_WK_DETECT_NET_SSD:
    {
        Result_SSD_Para_S stSSDParam = { 0 };

        getSSDResultPara(&stSSDParam);
        HI_U32 dst_score_size = stSSDParam.u32NumClasses*stSSDParam.u32KeepTopK * sizeof(HI_S32);
        HI_U32 dst_bbox_size = stSSDParam.u32NumClasses*stSSDParam.u32KeepTopK * 4 * sizeof(HI_S32);
        HI_U32 dst_roicnt_size = stSSDParam.u32NumClasses * sizeof(HI_S32);

        /////assit memory
        HI_U32 u32PrioSize = 0;
        for (HI_S32 i = 0; i < 6; i++)
        {
            u32PrioSize += SVP_GetPriorBoxSize(pstSSDParam, i);
        }
        HI_U32 u32SoftmaxSize = pstSSDParam->softmax_out_channel*pstSSDParam->softmax_out_height*pstSSDParam->softmax_out_width * sizeof(HI_S32);
        HI_U32 u32DetectionOutSize = SVP_GetDetectionOutSize(pstSSDParam);
        //HI_U32 assit_size = stSSDParam.u32TotlBoxNum * 16 * sizeof(HI_S32) + stSSDParam.u32TotlBoxNum * sizeof(SVP_SAMPLE_STACK_S);

        u32ResultMemSize = dst_score_size + dst_bbox_size + dst_roicnt_size + u32PrioSize + u32SoftmaxSize + u32DetectionOutSize;
        break;
    }
    default:
        break;
    }
    return u32ResultMemSize;
}

HI_S32 SvpSampleCnnDetectionOneSeg (const HI_CHAR *pszModelName, const HI_CHAR *paszPlicList[], const HI_U8 netType, HI_S32 s32Cnt)
{
    HI_S32 s32Ret = HI_SUCCESS;
    HI_U32 u32MaxInputNum = 10;
    HI_VOID *pExtraParam = NULL;

    SVP_NNIE_ONE_SEG_DET_S stDetParam = { 0 };
    SVP_NNIE_CFG_S     stDetCfg = { 0 };

    string strNetType = g_paszModelType_d[netType];;
    string strResultFolderDir = "result_" + strNetType + "/";
    _mkdir(strResultFolderDir.c_str());

    vector<SVP_SAMPLE_FILE_NAME_PAIR> imgNameRecoder;

    stDetCfg.pszModelName = pszModelName;
    memcpy(&stDetCfg.paszPicList, paszPlicList, sizeof(HI_VOID*)*s32Cnt);
    stDetCfg.u32MaxInputNum = u32MaxInputNum;
    stDetCfg.u32MaxBboxNum = 0;

    s32Ret = SvpSampleOneSegDetCnnInit(&stDetCfg, &stDetParam, netType);
    CHECK_EXP_RET(HI_SUCCESS != s32Ret, s32Ret, "SvpSampleOneSegCnnInit failed");

    /*****************menmory neeed by post-process of get detection result ****************/
    SVP_NNIE_SSD_S stSSDParam;
    if (SVP_SAMPLE_WK_DETECT_NET_SSD == netType)
    {
        pExtraParam = &stSSDParam;
        getSSDParm(&stSSDParam, &stDetParam);
    }
    HI_U32 u32ResultMemSize = SvpSampleGetResultMemSize(netType, &stSSDParam);
    if (u32ResultMemSize != 0) {
        stDetParam.ps32ResultMem = (HI_S32*)malloc(u32ResultMemSize);
        CHECK_EXP_GOTO(!stDetParam.ps32ResultMem, Fail, "Error: Malloc ps32ResultMem failed!");
    }

    HI_U32 u32Batch, u32LoopCnt, i, j, u32StartId;
    u32Batch = SVP_SAMPLE_MIN(stDetCfg.u32MaxInputNum, stDetParam.u32TotalImgNum);
    u32Batch = SVP_SAMPLE_MIN(u32Batch, stDetParam.astSrc[0].u32Num);
    CHECK_EXP_GOTO(0 == u32Batch, Fail2,
        "u32Batch = 0 failed! u32MaxInputNum(%d), tClfParam.u32TotalImgNum(%d), astSrc[0].u32Num(%d)",
        u32MaxInputNum, stDetParam.u32TotalImgNum, stDetParam.astSrc[0].u32Num);

    u32LoopCnt = stDetParam.u32TotalImgNum / u32Batch;

    for (i = 0, u32StartId = 0; i < u32LoopCnt; i++, u32StartId += u32Batch)
    {
        /* init vector mem with size u32Batch */
        imgNameRecoder.clear();

        s32Ret = SvpSampleReadAllSrcImg(stDetParam.fpSrc, stDetParam.astSrc, stDetParam.stModel.astSeg[0].u16SrcNum, imgNameRecoder);
        CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail, "Error(%#x):SvpSampleReadAllSrcImg failed!", s32Ret);

        CHECK_EXP_GOTO(imgNameRecoder.size() != u32Batch, Fail,
            "Error(%#x):imgNameRecoder.size(%d) != u32Batch(%d)", HI_FAILURE, (HI_U32)imgNameRecoder.size(), u32Batch);

        s32Ret = SvpSampleCnnDetectionForword(&stDetParam, &stDetCfg);
        CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail2, "SvpSampleCnnDetectionForword failed");

        s32Ret = SvpSampleDetOneSegGetResult(&stDetParam, netType, pExtraParam, strResultFolderDir, imgNameRecoder);
        CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail2, "SvpSampleDetOneSegGetResult failed");
    }

    u32Batch = stDetParam.u32TotalImgNum - u32StartId;
    if (u32Batch>0)
    {
        imgNameRecoder.clear();

        for (j = 0; j < stDetParam.stModel.astSeg[0].u16SrcNum; j++) {
            stDetParam.astSrc[j].u32Num = u32Batch;
        }
        for (j = 0; j < stDetParam.stModel.astSeg[0].u16DstNum; j++) {
            stDetParam.astDst[j].u32Num = u32Batch;
        }

        s32Ret = SvpSampleReadAllSrcImg(stDetParam.fpSrc, stDetParam.astSrc, stDetParam.stModel.astSeg[0].u16SrcNum, imgNameRecoder);
        CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail, "Error(%#x):SvpSampleReadAllSrcImg failed!", s32Ret);

        CHECK_EXP_GOTO(imgNameRecoder.size() != u32Batch, Fail,
            "Error(%#x):imgNameRecoder.size(%d) != u32Batch(%d)", HI_FAILURE, (HI_U32)imgNameRecoder.size(), u32Batch);

        s32Ret = SvpSampleCnnDetectionForword(&stDetParam, &stDetCfg);
        CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail2, "SvpSampleCnnDetectionForword failed");

        s32Ret = SvpSampleDetOneSegGetResult(&stDetParam, netType, pExtraParam, strResultFolderDir, imgNameRecoder);
        CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail2, "SvpSampleDetOneSegGetResult failed");
    }

Fail2:
    if (NULL != stDetParam.ps32ResultMem)
    {
        free(stDetParam.ps32ResultMem);
        stDetParam.ps32ResultMem = NULL;
    }
Fail:
    SvpSampleOneSegDetCnnDeinit(&stDetParam);
    return s32Ret;
}

void SvpSampleCnnDetSSD()
{
    printf("%s start ...\n", __FUNCTION__);
    SvpSampleCnnDetectionOneSeg(
        g_paszModelName_d[SVP_SAMPLE_WK_DETECT_NET_SSD],
        g_paszPicList_d[SVP_SAMPLE_WK_DETECT_NET_SSD],
        SVP_SAMPLE_WK_DETECT_NET_SSD);
    printf("%s end ...\n\n", __FUNCTION__);
    fflush(stdout);
}

void SvpSampleCnnDetYoloV1()
{
    printf("%s start ...\n", __FUNCTION__);
    SvpSampleCnnDetectionOneSeg(
        g_paszModelName_d[SVP_SAMPLE_WK_DETECT_NET_YOLOV1],
        g_paszPicList_d[SVP_SAMPLE_WK_DETECT_NET_YOLOV1],
        SVP_SAMPLE_WK_DETECT_NET_YOLOV1);
    printf("%s end ...\n\n", __FUNCTION__);
    fflush(stdout);
}

void SvpSampleCnnDetYoloV2()
{
    printf("%s start ...\n", __FUNCTION__);
    SvpSampleCnnDetectionOneSeg(
        g_paszModelName_d[SVP_SAMPLE_WK_DETECT_NET_YOLOV2],
        g_paszPicList_d[SVP_SAMPLE_WK_DETECT_NET_YOLOV2],
        SVP_SAMPLE_WK_DETECT_NET_YOLOV2);
    printf("%s end ...\n\n", __FUNCTION__);
    fflush(stdout);
}
