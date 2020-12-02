#include "rfcn_interface.h"
#include "mpi_nnie.h"
#include "SvpSampleWk.h"
#include "SvpSampleCom.h"
#ifdef USE_OPENCV
#include "cv_draw_rect.h"
#endif

const HI_CHAR *g_paszPicList_rfcn[][SVP_NNIE_MAX_INPUT_NUM] = {
    { "../data/detection/rfcn/resnet50/image_test_list.txt"}
};

#ifndef USE_FUNC_SIM /* inst wk */
const HI_CHAR *g_paszModelName_rfcn[] = {
    "../data/detection/rfcn/resnet50/inst/inst_rfcn_resnet50_inst.wk"
};
#else /* func wk */
const HI_CHAR *g_paszModelName_rfcn[] = {
    "../data/detection/rfcn/resnet50/inst/inst_rfcn_resnet50_func.wk"
};
#endif

/* the order is same with SVP_SAMPLE_WK_DETECT_NET_FASTER_RCNN_TYPE_E */
static const HI_CHAR *paszModelType_rfcn[] = {
    "SVP_SAMPLE_RFCN_RES50",
};

void setReportNodeInfo(NNIE_REPORT_NODE_INFO_S* pReportNodeInfo, const SVP_DST_BLOB_S* pReportBlob)
{
    pReportNodeInfo->u32ConvHeight = pReportBlob->unShape.stWhc.u32Height;
    pReportNodeInfo->u32ConvWidth  = pReportBlob->unShape.stWhc.u32Width;
    pReportNodeInfo->u32ConvMapNum = pReportBlob->unShape.stWhc.u32Chn;
    pReportNodeInfo->u32ConvStride = pReportBlob->u32Stride;
}

void getRFCNParam(stRFCNPara* para, SVP_NNIE_MULTI_SEG_S *wkParam)
{
    para->u32NumBeforeNms = 6000;

    para->model_info.enNetType = SVP_NNIE_NET_TYPE_ROI;

    //use dstNode para from wkParam
    //           stride
    //     0-0   208
    //     1-1   208
    //     2-3   208
    //     3-4   208
    //     4-5    96
    //     5-6    32
    setReportNodeInfo(para->model_info.astReportNodeInfo, wkParam->astDst);
    setReportNodeInfo(para->model_info.astReportNodeInfo + 1, wkParam->astDst + 1);
    setReportNodeInfo(para->model_info.astReportNodeInfo + 2, wkParam->astDst + 3);
    setReportNodeInfo(para->model_info.astReportNodeInfo + 3, wkParam->astDst + 4);
    setReportNodeInfo(para->model_info.astReportNodeInfo + 4, wkParam->astDst + 5);
    setReportNodeInfo(para->model_info.astReportNodeInfo + 5, wkParam->astDst + 6);

    para->model_info.u32ReportNodeNum = 6;

    para->model_info.u32MaxRoiFrameCnt = 300;
    para->model_info.u32MinSize = 16;
    para->model_info.u32SpatialScale = (HI_U32)(0.0625 * SVP_WK_QUANT_BASE);

    /* set anchors info */
    para->model_info.u32NumRatioAnchors = 3;
    para->model_info.u32NumScaleAnchors = 3;

    para->model_info.au32Ratios[0] = (HI_U32)(0.5 * SVP_WK_QUANT_BASE);
    para->model_info.au32Ratios[1] = (HI_U32)(1 * SVP_WK_QUANT_BASE);
    para->model_info.au32Ratios[2] = (HI_U32)(2 * SVP_WK_QUANT_BASE);

    para->model_info.au32Scales[0] = (HI_U32)(8 * SVP_WK_QUANT_BASE);
    para->model_info.au32Scales[1] = (HI_U32)(16 * SVP_WK_QUANT_BASE);
    para->model_info.au32Scales[2] = (HI_U32)(32 * SVP_WK_QUANT_BASE);

    para->model_info.u32ClassSize = 21;
    para->model_info.u32SrcHeight = wkParam->astSrc[0].unShape.stWhc.u32Height;
    para->model_info.u32SrcWidth  = wkParam->astSrc[0].unShape.stWhc.u32Width;

    para->data_size         = 21 * 7 * 7; //psroi cls output
    para->u32NmsThresh      = (HI_U32)(0.7 * SVP_WK_QUANT_BASE);
    para->u32ValidNmsThresh = (HI_U32)(0.3 * SVP_WK_QUANT_BASE);
    para->u32FilterThresh   = (HI_U32)(0 * SVP_WK_QUANT_BASE);
    para->u32ConfThresh     = (HI_U32)(0.3 * SVP_WK_QUANT_BASE);

}

HI_S32 SvpSampleWKRFCNRun(const HI_CHAR *pszModel, const HI_CHAR *pszPicList[],
        HI_U32 *pu32DstAlign, HI_S32 s32Cnt)
{
    HI_S32 s32Ret = HI_FAILURE;

    HI_BOOL bInstant = HI_TRUE;
    HI_U32 rois_num = 0;
    HI_U64 u64TempAddr1, u64TempAddr2;
    SVP_NNIE_HANDLE SvpNnieHandle;
    stRFCNPara para;

    HI_U32 assist_mem_size = 0;
    HI_U32 *assist_mem = NULL;
    HI_U32 length1 = 0;
    HI_U32 length2 = 0;
    HI_U32 length3 = 0;
    HI_U32* result1 = NULL;
    HI_U32* result2 = NULL;
    HI_U32* result3 = NULL;

    HI_U32 i = 0;

    SVP_NNIE_MULTI_SEG_S stDetParam = { 0 };
    SVP_NNIE_CFG_S stDetCfg = { 0 };
    std::vector<RFCN_BoxesInfo> vBoxesInfo;

    /* mkdir to save result, name folder by model type */
    string strNetType = paszModelType_rfcn[0];
    string strResultFolderDir = "result_" + strNetType + "/";
    _mkdir(strResultFolderDir.c_str());

    vector<SVP_SAMPLE_FILE_NAME_PAIR> imgNameRecoder;

#ifdef USE_OPENCV
    std::vector<SVPUtils_TaggedBox_S> vTaggedBoxes;
    string strBoxedImagePath;
#endif

    stDetCfg.pszModelName = pszModel;
    memcpy(stDetCfg.paszPicList, pszPicList, sizeof(HI_VOID*)*s32Cnt);
    stDetCfg.u32MaxInputNum = 10;
    stDetCfg.u32MaxBboxNum  = 300;

    s32Ret =  SvpSampleMultiSegCnnInit(&stDetCfg, &stDetParam, NULL, pu32DstAlign);
    CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail, "SvpSampleMultiSegCnnInit failed");

    getRFCNParam(&para, &stDetParam);

    //read data file or image file
    s32Ret = SvpSampleReadAllSrcImg(stDetParam.fpSrc, stDetParam.astSrc, stDetParam.stModel.astSeg[0].u16SrcNum, imgNameRecoder);
    CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail, "Error(%#x):SvpSampleReadAllSrcImg failed!", s32Ret);

    /* rfcn sample support 1 image input temp */
    CHECK_EXP_GOTO(imgNameRecoder.size() != 1, Fail,
        "Error(%#x):imgNameRecoder.size(%d) != 1", HI_FAILURE, (HI_U32)imgNameRecoder.size());

    // -------------------hardware part: first segment from rfcn.wk-------------------
    s32Ret = HI_MPI_SVP_NNIE_Forward(&SvpNnieHandle, stDetParam.astSrc, &stDetParam.stModel,
        stDetParam.astDst, &stDetParam.astCtrl[0], bInstant);
    CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail, "Error(%#x): RFCN Forward failed!", s32Ret);

    // calc rpn assist mem size and malloc assist memory
    assist_mem_size = GetRFCNAssistMemSize(&para);
    assist_mem = (HI_U32 *)malloc(assist_mem_size);
    CHECK_EXP_RET(NULL == assist_mem, HI_FAILURE, "Error(%#x): malloc_rpn_assist_mem_size failed!", s32Ret);
    memset(assist_mem, 0, assist_mem_size);


    // -------------------software part: gen rpn-------------------
    // astDst1[0]:rpn cls score, astDst1[1]:rpn bbox pred
    for (i = 0; i < stDetParam.stModel.astSeg[0].u16DstNum; i++)
    {
        para.model_info.astReportNodeInfo[i].u32ConvHeight = stDetParam.astDst[i].unShape.stWhc.u32Height;
        para.model_info.astReportNodeInfo[i].u32ConvMapNum = stDetParam.astDst[i].unShape.stWhc.u32Chn;
        para.model_info.astReportNodeInfo[i].u32ConvStride = stDetParam.astDst[i].u32Stride;
        para.model_info.astReportNodeInfo[i].u32ConvWidth  = stDetParam.astDst[i].unShape.stWhc.u32Width;
    }

    s32Ret = rfcn_rpn(&para,
        (HI_S32*)stDetParam.astDst[0].u64VirAddr,
        (HI_S32*)stDetParam.astDst[1].u64VirAddr,
        assist_mem,
        (HI_S32*)stDetParam.stRPN[0].u64PhyAddr,
        rois_num);
    CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail, "Error(%#x): RFCN rfcn_rpn failed!", s32Ret);

    // -------------------hardware part: psroi classification from the wk library-------------
    // pass the feature map from 1st segment's 4th report result to the 2nd segment's input.
    u64TempAddr1 = stDetParam.astSrc[1].u64VirAddr;
    stDetParam.astSrc[1].u64VirAddr = stDetParam.astDst[3].u64VirAddr;
    stDetParam.astSrc[1].u64PhyAddr = stDetParam.astDst[3].u64PhyAddr;

    s32Ret = HI_MPI_SVP_NNIE_ForwardWithBbox(&SvpNnieHandle, &stDetParam.astSrc[1], &stDetParam.stRPN[0], &stDetParam.stModel,
        &stDetParam.astDst[5], &stDetParam.astBboxCtrl[0], bInstant);
    CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail, "Error(%#x): RFCN HI_MPI_SVP_NNIE_ForwardWithBbox 1 failed!", s32Ret);

    // -------------------hardware part: psroi location from the wk library-------------------
    // pass the feature map from 1st segment's 5th report result to the 3rd segment's input.
    u64TempAddr2 = stDetParam.astSrc[2].u64VirAddr;
    stDetParam.astSrc[2].u64VirAddr = stDetParam.astDst[4].u64VirAddr;
    stDetParam.astSrc[2].u64PhyAddr = stDetParam.astDst[4].u64PhyAddr;

    s32Ret = HI_MPI_SVP_NNIE_ForwardWithBbox(&SvpNnieHandle, &stDetParam.astSrc[2], &stDetParam.stRPN[0], &stDetParam.stModel,
        &stDetParam.astDst[6], &stDetParam.astBboxCtrl[1], bInstant);
    CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail, "Error(%#x): RFCN HI_MPI_SVP_NNIE_ForwardWithBbox 2 failed!", s32Ret);

    para.model_info.u32ClassSize = stDetParam.astDst[5].unShape.stWhc.u32Width;

    /* result mem malloc */
    length1 = para.model_info.u32MaxRoiFrameCnt * para.model_info.u32ClassSize * sizeof(HI_U32);
    length2 = para.model_info.u32MaxRoiFrameCnt * para.model_info.u32ClassSize * SVP_WK_COORDI_NUM * sizeof(HI_U32);
    length3 = para.model_info.u32ClassSize * sizeof(HI_U32);

    result1 = (HI_U32*)malloc(sizeof(HI_U32) * length1);
    memset(result1, 0, sizeof(HI_U32) * length1);
    result2 = (HI_U32*)malloc(sizeof(HI_U32) * length2);
    memset(result2, 0, sizeof(HI_U32) * length2);
    result3 = (HI_U32*)malloc(sizeof(HI_U32) * length3);
    memset(result3, 0, sizeof(HI_U32) * length3);

    // 2nd param:rfcn_psroi_cls, 3rd param:rfcn_psroi_loc
    s32Ret = rfcn_detection_out(&para,
        (HI_U32*)stDetParam.astDst[5].u64PhyAddr,
        stDetParam.astDst[5].u32Stride,
        (HI_S32*)stDetParam.astDst[6].u64PhyAddr,
        stDetParam.astDst[6].u32Stride,
        (HI_S32*)stDetParam.stRPN[0].u64PhyAddr,
        rois_num,
        result1, &length1,
        result2, &length2,
        result3, &length3,
        assist_mem, vBoxesInfo,
        strResultFolderDir, imgNameRecoder[0]);
    CHECK_EXP_GOTO(HI_SUCCESS != s32Ret, Fail, "Error(%#x): rfcn_detection_out failed!", s32Ret);

#ifdef USE_OPENCV
    for (RFCN_BoxesInfo stBoxInfo : vBoxesInfo)
    {
        SVPUtils_TaggedBox_S stTaggedBox = {
            {stBoxInfo.u32XMin, stBoxInfo.u32YMin, stBoxInfo.u32XMax - stBoxInfo.u32XMin, stBoxInfo.u32YMax - stBoxInfo.u32YMin},
            stBoxInfo.u32Class,
            stBoxInfo.fScore
        };
        vTaggedBoxes.push_back(stTaggedBox);
    }
    strBoxedImagePath = strResultFolderDir + imgNameRecoder[0].first + "_det.png";
    SVPUtils_DrawBoxes(&stDetParam.astSrc[0], RGBPLANAR, strBoxedImagePath.c_str(), vTaggedBoxes);
#endif

    // restore stSrc so it could be free
    stDetParam.astSrc[1].u64VirAddr = u64TempAddr1;
    stDetParam.astSrc[1].u64PhyAddr = u64TempAddr1;
    stDetParam.astSrc[2].u64VirAddr = u64TempAddr2;
    stDetParam.astSrc[2].u64PhyAddr = u64TempAddr2;

Fail:
    free (assist_mem);
    free (result1);
    free (result2);
    free (result3);
    //SvpSampleWkDeinit(&stClfParam);
    SvpSampleMultiSegCnnDeinit(&stDetParam);
    return s32Ret;
}

void SvpSampleRoiDetRFCNResnet50()
{
    HI_U32 dstAlign[7] = {16,16,16,16,16,16,16};

    printf("%s start ...\n", __FUNCTION__);
    SvpSampleWKRFCNRun(
        g_paszModelName_rfcn[SVP_SAMPLE_WK_DETECT_NET_RFCN_RES50],
        g_paszPicList_rfcn[SVP_SAMPLE_WK_DETECT_NET_RFCN_RES50], dstAlign);
    printf("%s end ...\n\n", __FUNCTION__);
    fflush(stdout);
}
