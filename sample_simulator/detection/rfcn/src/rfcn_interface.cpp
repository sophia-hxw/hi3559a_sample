#include "rfcn_interface.h"

#include <math.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

#define RPN_NODE_NUM  (3)
#define RFCN_BACKGROUND_ID (0)

HI_S32 rfcn_rpn(
    stRFCNPara* para,

    HI_S32* ps32ClsScore,
    HI_S32* ps32BBoxPred,
    HI_U32* pu32MemPool,
    HI_S32* bottom_rois,
    HI_U32& rois_num)
{
    HI_U32 u32NmsThresh = para->u32NmsThresh;
    HI_U32 u32FilterThresh = para->u32FilterThresh;
    HI_S32 u32NumBeforeNms = para->u32NumBeforeNms;

    NNIE_MODEL_INFO_S model_info = para->model_info;

    return SvpDetRfcnRpn(
        ps32ClsScore,
        ps32BBoxPred,
        &model_info,
        u32NmsThresh,
        u32FilterThresh,
        u32NumBeforeNms,
        pu32MemPool,
        bottom_rois,
        &rois_num);
}

HI_S32 rfcn_detection_out(
    stRFCNPara* para,

    HI_S32* cls_prob_addr,                      //psroi cls output -> input
    HI_U32  cls_prob_stride,
    HI_S32* bbox_pred_addr,                     //psroi loc output -> input
    HI_U32  bbox_pred_stride,
    HI_S32* bottom_rois,
    HI_U32  rois_num,
    HI_U32* result_mem_score,  HI_U32 result_mem_size_score,
    HI_U32* result_mem_bbox,   HI_U32 result_mem_size_bbox,
    HI_U32* result_mem_roiout, HI_U32 result_mem_size_roiout,
    HI_U32* pu32MemPool,
    std::vector<RFCN_BoxesInfo> &vBoxesInfo,
    std::string& resultPath, SVP_SAMPLE_FILE_NAME_PAIR& imgNamePair)
{
    //get result
    SVP_SRC_MEM_INFO_S pstSrc = { 0 };
    pstSrc.u64VirAddr = (HI_U64)cls_prob_addr;
    pstSrc.u32Size = para->data_size * sizeof(HI_U32);

    SVP_SRC_MEM_INFO_S pstProp = { 0 };
    pstProp.u64VirAddr = (HI_U64)bottom_rois;
    pstProp.u32Size = para->model_info.u32MaxRoiFrameCnt * SVP_WK_COORDI_NUM * sizeof(HI_U32);

    SVP_SRC_MEM_INFO_S pstDstScore = { 0 };
    pstDstScore.u64VirAddr = (HI_U64)result_mem_score;
    pstDstScore.u32Size = result_mem_size_score * sizeof(HI_U32);

    SVP_SRC_MEM_INFO_S pstDstBbox = { 0 };
    pstDstBbox.u64VirAddr = (HI_U64)result_mem_bbox;
    pstDstBbox.u32Size = result_mem_size_bbox * sizeof(HI_U32);

    SVP_SRC_MEM_INFO_S pstRoiOutCnt = { 0 };
    pstRoiOutCnt.u64VirAddr = (HI_U64)result_mem_roiout;
    pstRoiOutCnt.u32Size = result_mem_size_roiout * sizeof(HI_U32);

    SVP_SRC_MEM_INFO_S pstMemPool = { 0 };
    pstMemPool.u64VirAddr = (HI_U64)pu32MemPool;
    pstMemPool.u32Size = para->model_info.u32MaxRoiFrameCnt * para->model_info.u32ClassSize +
                         para->model_info.u32MaxRoiFrameCnt * SVP_WK_COORDI_NUM +
                         para->model_info.u32MaxRoiFrameCnt * SVP_WK_PROPOSAL_WIDTH;

    HI_U32 u32FcScoreStride = cls_prob_stride;
    HI_U32 u32FcBboxStride = bbox_pred_stride;

    SvpDetRfcnGetResult(
        &pstSrc, u32FcScoreStride, bbox_pred_addr,
        u32FcBboxStride, &pstProp, rois_num,
        &pstDstScore, &pstDstBbox,
        &pstRoiOutCnt,
        &(para->model_info),
        para->u32ConfThresh,
        para->u32ValidNmsThresh,
        &pstMemPool,
        vBoxesInfo, resultPath, imgNamePair);

    return HI_SUCCESS;
}

HI_U32 SvpDetRfcnGetAssistMemSize(const stRFCNPara* para)
{
    HI_U32 u32NumAnchors = (para->model_info.u32NumRatioAnchors) *
                           (para->model_info.u32NumScaleAnchors)*
                           (para->model_info.astReportNodeInfo[0].u32ConvHeight) *
                           (para->model_info.astReportNodeInfo[0].u32ConvWidth);

    HI_U32 u32AnchorSize    = u32NumAnchors* SVP_WK_COORDI_NUM * sizeof(HI_U32);
    HI_U32 u32BboxDeltaSize = u32AnchorSize;
    HI_U32 u32ProposalSize  = u32NumAnchors * SVP_WK_PROPOSAL_WIDTH * sizeof(HI_U32);
    HI_U32 u32RatioSize     = (para->model_info.u32NumRatioAnchors) * SVP_WK_COORDI_NUM * sizeof(HI_FLOAT);
    HI_U32 u32ScaleSize     = (para->model_info.u32NumRatioAnchors) *
                              (para->model_info.u32NumScaleAnchors) * SVP_WK_COORDI_NUM * sizeof(HI_FLOAT);
    HI_U32 u32ScoresSize    = u32NumAnchors * SVP_WK_SCORE_NUM * sizeof(HI_FLOAT);
    HI_U32 u32StackSize     = MAX_STACK_DEPTH * sizeof(SVP_SAMPLE_STACK_S);

    HI_U32 u32TotalSize = u32AnchorSize + u32BboxDeltaSize + u32ProposalSize + u32RatioSize +
                          u32ScaleSize + u32ScoresSize + u32StackSize;

    return u32TotalSize;
}

HI_S32 SvpDetRfcnRpn(
    HI_S32 *ps32ClsScore,
    HI_S32 *ps32BBoxPred,
    NNIE_MODEL_INFO_S* pstModelInfo,
    HI_U32 u32NmsThresh,
    HI_U32 u32FilterThresh,
    HI_U32 u32NumBeforeNms,
    HI_U32 *pu32MemPool,
    HI_S32 *ps32ProposalResult,
    HI_U32 *pu32NumRois)
{
    printf("RPN function begins\n");

    NNIE_REPORT_NODE_INFO_S conv[RPN_NODE_NUM] = { 0 };
    memset(conv, 0, sizeof(NNIE_REPORT_NODE_INFO_S) * RPN_NODE_NUM);

    HI_S32 s32Ret = HI_FAILURE;

    /******************** Get parameters from Model and Config ***********************/
    HI_U32 u32OriImHeight = pstModelInfo->u32SrcHeight;
    HI_U32 u32OriImWidth = pstModelInfo->u32SrcWidth;

    if (SVP_NNIE_NET_TYPE_ROI != pstModelInfo->enNetType)
    {
        for (HI_U32 i = 0; i < RPN_NODE_NUM; i++)
        {
            conv[i].u32ConvHeight = pstModelInfo->astReportNodeInfo[i].u32ConvHeight;
            conv[i].u32ConvWidth = pstModelInfo->astReportNodeInfo[i].u32ConvWidth;
            conv[i].u32ConvMapNum = pstModelInfo->astReportNodeInfo[i].u32ConvMapNum;
            conv[i].u32ConvStride = pstModelInfo->astReportNodeInfo[i].u32ConvStride;
        }
    }
    else
    {
        for (HI_U32 i = 0; i < RPN_NODE_NUM - 1; i++)
        {
            conv[i + 1].u32ConvHeight = pstModelInfo->astReportNodeInfo[i].u32ConvHeight;
            conv[i + 1].u32ConvWidth = pstModelInfo->astReportNodeInfo[i].u32ConvWidth;
            conv[i + 1].u32ConvMapNum = pstModelInfo->astReportNodeInfo[i].u32ConvMapNum;
            conv[i + 1].u32ConvStride = pstModelInfo->astReportNodeInfo[i].u32ConvStride;
        }
    }

    HI_U32 u32MaxRois = pstModelInfo->u32MaxRoiFrameCnt;

    HI_U32 au32BaseAnchor[SVP_WK_COORDI_NUM] = { 0, 0, (pstModelInfo->u32MinSize - 1), (pstModelInfo->u32MinSize - 1) };

    /*********************************** Faster RCNN *********************************************/
    /********* calculate the start pointer of each part in MemPool *********/
    HI_U32* pu32Ptr = pu32MemPool;

    /* base RatioAnchors and ScaleAnchors */
    HI_S32* ps32Anchors = (HI_S32*)pu32Ptr;

    HI_U32 u32NumAnchors = (pstModelInfo->u32NumRatioAnchors) *
                           (pstModelInfo->u32NumScaleAnchors) *
                           (conv[1].u32ConvHeight * conv[1].u32ConvWidth);

    HI_U32 u32Size = 0;
    u32Size = SVP_WK_COORDI_NUM * u32NumAnchors;
    pu32Ptr += u32Size;

    /* BboxDelta */
    HI_S32* ps32BboxDelta = (HI_S32*)pu32Ptr;
    pu32Ptr += u32Size;

    /* Proposal info */
    HI_S32* ps32Proposals = (HI_S32*)pu32Ptr;
    u32Size = SVP_WK_PROPOSAL_WIDTH * u32NumAnchors;
    pu32Ptr += u32Size;

    /* RatioAnchors and ScaleAnchors info */
    HI_FLOAT* pf32RatioAnchors = (HI_FLOAT*)pu32Ptr;

    HI_FLOAT* pf32Ptr = pf32RatioAnchors;
    u32Size = pstModelInfo->u32NumRatioAnchors * SVP_WK_COORDI_NUM;
    pf32Ptr += u32Size;

    HI_FLOAT* pf32ScaleAnchors = pf32Ptr;
    u32Size = pstModelInfo->u32NumScaleAnchors * pstModelInfo->u32NumRatioAnchors * SVP_WK_COORDI_NUM;
    pf32Ptr += u32Size;

    /* Proposal scores */
    HI_FLOAT* pf32Scores = pf32Ptr;
    u32Size = u32NumAnchors * SVP_WK_SCORE_NUM;
    pf32Ptr += u32Size;

    /* quick sort Stack */
    SVP_SAMPLE_STACK_S* pstStack = (SVP_SAMPLE_STACK_S*)pf32Ptr;

    /********************* Generate the base anchor ***********************/
    s32Ret = GenBaseAnchor(pf32RatioAnchors, pstModelInfo->au32Ratios, pstModelInfo->u32NumRatioAnchors,
        pf32ScaleAnchors, pstModelInfo->au32Scales, pstModelInfo->u32NumScaleAnchors, au32BaseAnchor);
    SVP_CHECK(HI_SUCCESS == s32Ret, HI_FAILURE);

    /******************* Copy the anchors to every pixel in the feature map ******************/
    s32Ret = SetAnchorInPixel(ps32Anchors,
        pf32ScaleAnchors,
        conv[1].u32ConvHeight,
        conv[1].u32ConvWidth,
        pstModelInfo->u32NumScaleAnchors * pstModelInfo->u32NumRatioAnchors,
        pstModelInfo->u32SpatialScale);
    SVP_CHECK(HI_SUCCESS == s32Ret, HI_FAILURE);

    /********** do transpose, convert the blob from (M,C,H,W) to (M,H,W,C) **********/
    HI_U32 u32MapSize = (conv[2].u32ConvHeight) * (conv[2].u32ConvStride / sizeof(HI_U32));
    HI_U32 u32AnchorsPerPixel = pstModelInfo->u32NumRatioAnchors * pstModelInfo->u32NumScaleAnchors;
    HI_U32 u32BgBlobSize = u32AnchorsPerPixel * u32MapSize;
    HI_U32 u32LineSize = (conv[2].u32ConvStride) / sizeof(HI_U32);

    for (HI_U32 c = 0; c < conv[2].u32ConvMapNum; c++)
    {
        for (HI_U32 h = 0; h < conv[2].u32ConvHeight; h++)
        {
            for (HI_U32 w = 0; w < conv[2].u32ConvWidth; w++)
            {
                HI_U32 u32SrcBboxIndex = c * u32MapSize + h * u32LineSize + w;
                HI_U32 u32SrcBgProbIndex = (c / SVP_WK_COORDI_NUM) * u32MapSize + h * u32LineSize + w;
                HI_U32 u32SrcFgProbIndex = u32BgBlobSize + u32SrcBgProbIndex;

                HI_U32 u32DesBox = (u32AnchorsPerPixel)* (h * conv[2].u32ConvWidth + w) + (c / SVP_WK_COORDI_NUM);
                HI_U32 u32DesBboxDeltaIndex = SVP_WK_COORDI_NUM * u32DesBox + (c % SVP_WK_COORDI_NUM);

                ps32BboxDelta[u32DesBboxDeltaIndex] = ps32BBoxPred[u32SrcBboxIndex];

                HI_U32 u32DesScoreIndex = SVP_WK_SCORE_NUM * u32DesBox;
                pf32Scores[u32DesScoreIndex + 0] = (HI_FLOAT)ps32ClsScore[u32SrcBgProbIndex] / SVP_WK_QUANT_BASE;
                pf32Scores[u32DesScoreIndex + 1] = (HI_FLOAT)ps32ClsScore[u32SrcFgProbIndex] / SVP_WK_QUANT_BASE;
            }
        }
    }

    /************************* do softmax ****************************/
    s32Ret = SoftMax_N(pf32Scores, SVP_WK_SCORE_NUM, u32NumAnchors);
    SVP_CHECK(HI_SUCCESS == s32Ret, HI_FAILURE);

    /************************* BBox Transform *****************************/
    s32Ret = BboxTransform_N(ps32Proposals, ps32Anchors, ps32BboxDelta, pf32Scores, u32NumAnchors);
    SVP_CHECK(HI_SUCCESS == s32Ret, HI_FAILURE);

    /************************ clip bbox *****************************/
    s32Ret = BboxClip_N(ps32Proposals, u32OriImWidth, u32OriImHeight, u32NumAnchors);
    SVP_CHECK(HI_SUCCESS == s32Ret, HI_FAILURE);

    /************ remove the bboxes which are too small ***********/
    s32Ret = BboxSmallSizeFilter_N(ps32Proposals, pstModelInfo->u32MinSize, pstModelInfo->u32MinSize, u32NumAnchors);
    SVP_CHECK(HI_SUCCESS == s32Ret, HI_FAILURE);

    /********** remove low score bboxes ************/
    HI_U32 u32NumAfterFilter = 0;
    s32Ret = FilterLowScoreBbox(ps32Proposals, u32NumAnchors, u32NmsThresh, u32FilterThresh, &u32NumAfterFilter);
    SVP_CHECK(HI_SUCCESS == s32Ret, HI_FAILURE);

    /********** sort ***********/
    s32Ret = NonRecursiveArgQuickSort(ps32Proposals, 0, (HI_S32)u32NumAfterFilter - 1, pstStack);
    SVP_CHECK(HI_SUCCESS == s32Ret, HI_FAILURE);

    u32NumAfterFilter = SVP_MIN(u32NumAfterFilter, u32NumBeforeNms);

    /****************** write Proposal_before_nms_rpn *********************/
    s32Ret = dumpProposal(ps32Proposals, "Proposal_before_nms_rpn.txt", u32NumAnchors);
    SVP_CHECK(HI_SUCCESS == s32Ret, HI_FAILURE);

    /* do nms to remove highly overlapped bbox */
    s32Ret = NonMaxSuppression(ps32Proposals, u32NumAfterFilter, u32NmsThresh);
    SVP_CHECK(HI_SUCCESS == s32Ret, HI_FAILURE);

    /****************** write Proposal_after_nms_rpn *********************/

    s32Ret = dumpProposal(ps32Proposals, "Proposal_after_nms_rpn.txt", u32NumAnchors);
    SVP_CHECK(HI_SUCCESS == s32Ret, HI_FAILURE);

    /************** write the final result to output ***************/
    s32Ret = getRPNresult(ps32ProposalResult, pu32NumRois, u32MaxRois, ps32Proposals, u32NumAfterFilter);
    SVP_CHECK(HI_SUCCESS == s32Ret, HI_FAILURE);

    printf("RPN function complete\n");
    return HI_SUCCESS;
    /******************** end of FasterRCNN RPN **********************/
}

HI_S32 SvpDetRfcnGetResult(SVP_SRC_MEM_INFO_S *pstSrc, HI_U32 u32FcScoreStride, HI_S32* ps32FcBbox, HI_U32 u32FcBboxStride,
    SVP_SRC_MEM_INFO_S *pstProposal, HI_U32 u32RoiCnt,
    SVP_DST_MEM_INFO_S *pstDstScore, SVP_DST_MEM_INFO_S *pstDstBbox,
    SVP_MEM_INFO_S *pstRoiOutCnt, NNIE_MODEL_INFO_S *pstModelInfo,
    HI_U32 u32ConfThresh, HI_U32 u32NmsThresh, SVP_MEM_INFO_S *pstMemPool, std::vector<RFCN_BoxesInfo> &vBoxesInfo,
    std::string& resultPath, SVP_SAMPLE_FILE_NAME_PAIR& imgNamePair)
{
    /********* check the proposal size *********/
    //...

    /************* define variables *****************/
    HI_S32 s32Ret = HI_FAILURE;
    HI_S32* ps32Ptr = NULL;
    HI_U32 u32Size = 0;

    /******************* Get or calculate parameters **********************/
    HI_U32  u32BackGroundId = RFCN_BACKGROUND_ID;

    HI_U32 u32MaxRoi = pstModelInfo->u32MaxRoiFrameCnt;

    HI_U32 u32ClassNum = pstModelInfo->u32ClassSize;   /*channel num is equal to class size, cls_score class*/

    u32Size = u32RoiCnt*u32ClassNum;              /*s32Height*s32Width;*/

    HI_U32 u32FcScoreWidth = u32FcScoreStride / sizeof(HI_U32);
    HI_U32 u32FcBboxWidth = u32FcBboxStride / sizeof(HI_U32);

    HI_S32* ps32FcScore = (HI_S32*)pstSrc->u64VirAddr;  /* Pointer to scores (ave pooling) */
    u32Size = u32MaxRoi * u32FcScoreWidth;
    HI_U32 u32OriImWidth = pstModelInfo->u32SrcWidth;
    HI_U32 u32OriImHeight = pstModelInfo->u32SrcHeight;

    /*************** Get Start Pointer of MemPool ******************/
    HI_FLOAT* pf32FcScoresMemPool = (HI_FLOAT*)(pstMemPool->u64VirAddr);
    HI_FLOAT* pf32Ptr = pf32FcScoresMemPool;

    u32Size = u32MaxRoi * u32ClassNum;
    pf32Ptr += u32Size;

    HI_S32* ps32FcBboxMemPool = (HI_S32*)pf32Ptr;
    ps32Ptr = (HI_S32*)pf32Ptr;
    u32Size = u32MaxRoi * SVP_WK_COORDI_NUM;
    ps32Ptr += u32Size;

    HI_S32* ps32ProposalMemPool = (HI_S32*)ps32Ptr;
    ps32Ptr = ps32ProposalMemPool;
    u32Size = u32MaxRoi * SVP_WK_PROPOSAL_WIDTH;
    ps32Ptr += u32Size;
    SVP_SAMPLE_STACK_S* pstStack = (SVP_SAMPLE_STACK_S*)ps32Ptr;

    // prepare input data
    // change score output to s32 and set to pf32FcScoresMemPool
    for (HI_U32 u32roiIdx = 0; u32roiIdx < u32RoiCnt; u32roiIdx++)
    {
        for (HI_U32 u32classIdx = 0; u32classIdx < u32ClassNum; u32classIdx++)
        {
            HI_U32 u32SrcIndex = u32classIdx + u32roiIdx * u32FcScoreWidth;
            HI_U32 u32DstIndex = u32classIdx + u32roiIdx * u32ClassNum;

            pf32FcScoresMemPool[u32DstIndex] = (HI_FLOAT)ps32FcScore[u32SrcIndex] / SVP_WK_QUANT_BASE;
        }
    }

    // set delta output to ps32FcBboxMemPool
    for (HI_U32 u32roiIdx = 0; u32roiIdx < u32RoiCnt; u32roiIdx++)
    {
        for (HI_U32 u32CoordIdx = 0; u32CoordIdx < SVP_WK_COORDI_NUM; u32CoordIdx++)
        {
            HI_U32 u32SrcIndex = u32CoordIdx + u32roiIdx * u32FcBboxWidth + SVP_WK_COORDI_NUM;
            HI_U32 u32DstIndex = u32CoordIdx + u32roiIdx * SVP_WK_COORDI_NUM;
            ps32FcBboxMemPool[u32DstIndex] = ps32FcBbox[u32SrcIndex];
        }
    }

#if 0
    FILE* fc_bbox = SvpDetOpenFile((resultPath + imgNamePair.first + "_fc_bbox.txt").c_str(), "w");
    SVP_CHECK(NULL != fc_bbox, HI_FAILURE);

    for (HI_U32 i = 0; i < u32RoiCnt; i++)
    {
        for (HI_U32 j = 0; j < SVP_WK_COORDI_NUM; j++)
        {
            fprintf(fc_bbox, "%d\n", ps32FcBbox[i*SVP_WK_COORDI_NUM + j]);
        }
    }
    SvpDetCloseFile(fc_bbox);
    fc_bbox = NULL;
#endif

    /************** bbox transform ************
    change the fc output to Proposal temp MemPool.
    Each Line of the Proposal has 6 bits.
    The Format of the Proposal is:
    0-3: The four coordinate of the bbox, x1,y1,x2,y2
    4: The Confidence Score of the bbox
    5: The suppressed flag
    ******************************************/

    stringstream ss;

    string pre_tmp = "roi_pos_out_";
    string post_tmp = ".txt";
    string fileName = resultPath + imgNamePair.first + "_" + pre_tmp + "ALL" + post_tmp;
    ofstream foutALL(fileName.c_str());
    SVP_CHECK(foutALL.good(), HI_FAILURE);

    HI_BOOL hasResultOut = HI_FALSE;

    PrintBreakLine(HI_TRUE);

    ss << u32OriImWidth << "  " << u32OriImHeight << endl;
    cout << ss.str();
    foutALL << ss.str();
    ss.str("");

    HI_S32* ps32Proposals = (HI_S32*)pstProposal->u64VirAddr;

    for (HI_U32 u32classIdx = 0; u32classIdx< u32ClassNum; u32classIdx++)
    {
        for (HI_U32 u32roiIdx = 0; u32roiIdx < u32RoiCnt; u32roiIdx++)
        {
            HI_FLOAT pf32Proposals[SVP_WK_PROPOSAL_WIDTH] = { 0.0f };
            HI_FLOAT pf32Anchors[SVP_WK_COORDI_NUM]       = { 0.0f };
            HI_FLOAT pf32BboxDelta[SVP_WK_COORDI_NUM]     = { 0.0f };

            // init anchor delta and score
            // score use 20.12 and coords not use
            for (HI_U32 k = 0; k < SVP_WK_COORDI_NUM; k++)
            {
                HI_U32 u32index = SVP_WK_COORDI_NUM*u32roiIdx + k;
                pf32Anchors[k] = (HI_FLOAT)ps32Proposals[u32index] / SVP_WK_QUANT_BASE;
                pf32BboxDelta[k] = (HI_FLOAT)ps32FcBboxMemPool[u32index];
            }

            HI_FLOAT f32FcScores = pf32FcScoresMemPool[u32ClassNum * u32roiIdx + u32classIdx];

            s32Ret = BboxTransform_FLOAT(pf32Proposals, pf32Anchors, pf32BboxDelta, f32FcScores);
            SVP_CHECK(HI_SUCCESS == s32Ret, HI_FAILURE);

            for (HI_U32 k = 0; k < SVP_WK_PROPOSAL_WIDTH; k++) {
                ps32ProposalMemPool[u32roiIdx*SVP_WK_PROPOSAL_WIDTH + k] = (HI_S32)pf32Proposals[k];
            }
        }

        /* clip bbox */
        s32Ret = BboxClip_N(ps32ProposalMemPool, u32OriImWidth, u32OriImHeight, u32RoiCnt);
        SVP_CHECK(HI_SUCCESS == s32Ret, HI_FAILURE);

        HI_S32* ps32ProposalTmp = ps32ProposalMemPool;

        /* sort */
        s32Ret = NonRecursiveArgQuickSort(ps32ProposalTmp, 0, (HI_S32)u32RoiCnt - 1, pstStack);
        SVP_CHECK(HI_SUCCESS == s32Ret, HI_FAILURE);

        /* NMS */
        s32Ret = NonMaxSuppression(ps32ProposalTmp, u32RoiCnt, u32NmsThresh);
        SVP_CHECK(HI_SUCCESS == s32Ret, HI_FAILURE);

        pre_tmp = "roi_pos_out_";
        post_tmp = ".txt";
        fileName = resultPath + imgNamePair.first + "_" + pre_tmp + to_string((HI_U64)u32classIdx) + post_tmp;
        ofstream fout(fileName.c_str());
        SVP_CHECK(fout.good(), HI_FAILURE);

        HI_U32 u32RoiOutCnt = 0;
        HI_S32* ps32DstScore  = (HI_S32*)pstDstScore->u64VirAddr;
        HI_S32* ps32DstBbox   = (HI_S32*)pstDstBbox->u64VirAddr;
        HI_S32* ps32RoiOutCnt = (HI_S32*)pstRoiOutCnt->u64VirAddr;

        ps32DstScore += (HI_S32)(u32classIdx * u32MaxRoi);
        ps32DstBbox  += (HI_S32)(u32classIdx * u32MaxRoi * SVP_WK_COORDI_NUM);

        for (HI_U32 u32roiIdx = 0; u32roiIdx < u32RoiCnt; u32roiIdx++)
        {
            HI_U32 u32ProposalMemPoolIndex = SVP_WK_PROPOSAL_WIDTH * u32roiIdx;

            if (RPN_SUPPRESS_FALSE == ps32ProposalMemPool[u32ProposalMemPoolIndex + 5] &&
                ps32ProposalMemPool[u32ProposalMemPoolIndex + 4] > (HI_S32)u32ConfThresh)
            {
                HI_U32 u32RoiCoordIndex = u32RoiOutCnt * SVP_WK_COORDI_NUM;

                ps32DstBbox[u32RoiCoordIndex + 0] = ps32ProposalMemPool[u32ProposalMemPoolIndex + 0];
                ps32DstBbox[u32RoiCoordIndex + 1] = ps32ProposalMemPool[u32ProposalMemPoolIndex + 1];
                ps32DstBbox[u32RoiCoordIndex + 2] = ps32ProposalMemPool[u32ProposalMemPoolIndex + 2];
                ps32DstBbox[u32RoiCoordIndex + 3] = ps32ProposalMemPool[u32ProposalMemPoolIndex + 3];

                ps32DstScore[u32RoiOutCnt] = ps32ProposalMemPool[u32ProposalMemPoolIndex + 4];  // score

                HI_FLOAT f32ScoreOut = (HI_FLOAT)ps32DstScore[u32RoiOutCnt] / SVP_WK_QUANT_BASE;

                if (f32ScoreOut > 1.0 || f32ScoreOut < 0.0) {
                    printf("ERROR in f32ScoreOut(%f), out of range[0,1]\n", f32ScoreOut);
                }
                else {
                    HI_S32 s32XMin = ps32DstBbox[u32RoiCoordIndex + 0];
                    HI_S32 s32YMin = ps32DstBbox[u32RoiCoordIndex + 1];
                    HI_S32 s32XMax = ps32DstBbox[u32RoiCoordIndex + 2];
                    HI_S32 s32YMax = ps32DstBbox[u32RoiCoordIndex + 3];

                    ss << imgNamePair.first << "  " <<
                        setw(4) << u32classIdx << "  " <<
                        fixed << setprecision(8) << f32ScoreOut << "  " <<
                        setw(4) << s32XMin << "  " <<
                        setw(4) << s32YMin << "  " <<
                        setw(4) << s32XMax << "  " <<
                        setw(4) << s32YMax << endl;

                    hasResultOut = HI_TRUE;
                    /* dump to roi_pos_out_ALL.txt except class 0*/
                    if (u32BackGroundId != u32classIdx) {
                        foutALL << ss.str();
                        RFCN_BoxesInfo stBoxInfo = { (HI_U32)s32XMin, (HI_U32)s32YMin, (HI_U32)s32XMax, (HI_U32)s32YMax, u32classIdx, f32ScoreOut };
                        vBoxesInfo.push_back(stBoxInfo);
                        cout << ss.str();
                    }

                    /* dump to roi_pos_out_i.txt */
                    fout << ss.str();
                    ss.str("");
                }

                u32RoiOutCnt++;
            }

            if (u32RoiOutCnt >= u32RoiCnt)
                break;
        }

        ps32RoiOutCnt[u32classIdx] = (HI_S32)u32RoiOutCnt;

        fout.close();

        PrintBreakLine(hasResultOut);
        hasResultOut = HI_FALSE;
    }
    foutALL.close();

    return HI_SUCCESS;
}

