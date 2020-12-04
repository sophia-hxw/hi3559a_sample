#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include "hi_nnie.h"
#include "rfcn_interface.h"

#define RPN_NODE_NUM  (3)

HI_S32 HI_MPI_SVP_NNIE_WK_CNN_FASTER_RPN_Ref(
    HI_S32 *ps32Src,
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

    /******************** define parameters ****************/
    HI_U32 u32Size              = 0;
    HI_S32* ps32Anchors         = NULL;
    HI_S32* ps32BboxDelta       = NULL;
    HI_S32* ps32Proposals       = NULL;
    HI_U32 u32OriImHeight       = 0;
    HI_U32 u32OriImWidth        = 0;
    HI_U32* pu32Ptr             = NULL;

    HI_U32 u32MaxRois           = 0;
    HI_U32 u32NumAfterFilter    = 0;
    HI_U32 u32NumAnchors        = 0;
    HI_FLOAT* pf32RatioAnchors  = NULL;
    HI_FLOAT* pf32Ptr           = NULL;
    HI_FLOAT* pf32ScaleAnchors  = NULL;
    HI_FLOAT* pf32Scores        = NULL;

    HI_U32 u32SrcBboxIndex      = 0;
    HI_U32 u32SrcFgProbIndex    = 0;
    HI_U32 u32SrcBgProbIndex    = 0;

    HI_U32 u32SrcProbBias       = 0;
    HI_U32 u32DesBox            = 0;
    HI_U32 u32BgBlobSize        = 0;
    HI_U32 u32AnchorsPerPixel   = 0;
    HI_U32 u32MapSize           = 0;
    HI_U32 u32LineSize          = 0;

    HI_U32 u32DesBboxDeltaIndex = 0;
    HI_U32 u32DesScoreIndex     = 0;

    NNIE_STACK_S* pstStack      = NULL;

    NNIE_REPORT_NODE_INFO_S conv[RPN_NODE_NUM];
    memset(conv, 0, sizeof(NNIE_REPORT_NODE_INFO_S) * RPN_NODE_NUM);

    HI_S32 s32Ret               = HI_FAILURE;

    /******************** Get parameters from Model and Config ***********************/
    u32OriImHeight = pstModelInfo->u32SrcHeight;
    u32OriImWidth = pstModelInfo->u32SrcWidth;

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

    u32MaxRois = pstModelInfo->u32MaxRoiFrameCnt;

    HI_U32 au32BaseAnchor[SVP_WK_COORDI_NUM] = {0, 0, (pstModelInfo->u32MinSize -1), (pstModelInfo->u32MinSize -1)};

    /*********************************** Faster RCNN *********************************************/
    /********* calculate the start pointer of each part in MemPool *********/
    /* base RatioAnchors and ScaleAnchors */
    pu32Ptr = (HI_U32*)pu32MemPool;
    ps32Anchors = (HI_S32*)pu32Ptr;
    u32NumAnchors = (pstModelInfo->u32NumRatioAnchors) *
                    (pstModelInfo->u32NumScaleAnchors) *
                    (conv[1].u32ConvHeight * conv[1].u32ConvWidth);
    u32Size = SVP_WK_COORDI_NUM * u32NumAnchors;
    pu32Ptr += u32Size;

    /* BboxDelta */
    ps32BboxDelta = (HI_S32*)pu32Ptr;
    pu32Ptr += u32Size;

    /* Proposal info */
    ps32Proposals = (HI_S32*)pu32Ptr;
    u32Size = SVP_WK_PROPOSAL_WIDTH * u32NumAnchors;
    pu32Ptr += u32Size;

    /* RatioAnchors and ScaleAnchors info */
    pf32RatioAnchors = (HI_FLOAT*)pu32Ptr;
    pf32Ptr = (HI_FLOAT*)pu32Ptr;
    u32Size = pstModelInfo->u32NumRatioAnchors * SVP_WK_COORDI_NUM;
    pf32Ptr += u32Size;

    pf32ScaleAnchors = pf32Ptr;
    u32Size = pstModelInfo->u32NumScaleAnchors * pstModelInfo->u32NumRatioAnchors * SVP_WK_COORDI_NUM;
    pf32Ptr += u32Size;

    /* Proposal scores */
    pf32Scores = pf32Ptr;
    u32Size = u32NumAnchors * SVP_WK_SCORE_NUM;
    pf32Ptr += u32Size;

    /* quick sort Stack */
    pstStack = (NNIE_STACK_S*)pf32Ptr;

    /********************* Generate the base anchor ***********************/
    s32Ret = GenBaseAnchor(pf32RatioAnchors, pstModelInfo->au32Ratios, pstModelInfo->u32NumRatioAnchors,
                           pf32ScaleAnchors, pstModelInfo->au32Scales, pstModelInfo->u32NumScaleAnchors,
                           au32BaseAnchor);
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
    u32MapSize = (conv[2].u32ConvHeight) * (conv[2].u32ConvStride / sizeof(HI_U32));
    u32AnchorsPerPixel = pstModelInfo->u32NumRatioAnchors * pstModelInfo->u32NumScaleAnchors;
    u32BgBlobSize = u32AnchorsPerPixel * u32MapSize;
    u32LineSize = (conv[2].u32ConvStride) / sizeof(HI_U32);

    u32SrcProbBias = (conv[0].u32ConvMapNum ) *
                     (conv[0].u32ConvHeight) *
                     (conv[1].u32ConvStride / sizeof(HI_U32)); /* skip the 1st conv */

    for (HI_U32 c = 0; c < conv[2].u32ConvMapNum; c++)
    {
        for (HI_U32 h = 0; h < conv[2].u32ConvHeight; h++)
        {
            for (HI_U32 w = 0; w < conv[2].u32ConvWidth; w++)
            {
                u32SrcBgProbIndex = u32SrcProbBias + (c / SVP_WK_COORDI_NUM) * u32MapSize + h * u32LineSize + w;
                u32SrcBboxIndex = c * u32MapSize + h * u32LineSize + w;
                u32SrcBgProbIndex = (c / SVP_WK_COORDI_NUM) * u32MapSize + h * u32LineSize + w;
                u32SrcFgProbIndex = u32BgBlobSize + u32SrcBgProbIndex;

                u32DesBox = (u32AnchorsPerPixel) * (h * conv[2].u32ConvWidth + w) + (c / SVP_WK_COORDI_NUM);

                u32DesBboxDeltaIndex = SVP_WK_COORDI_NUM * u32DesBox + (c % SVP_WK_COORDI_NUM);
                ps32BboxDelta[u32DesBboxDeltaIndex] = ps32BBoxPred[u32SrcBboxIndex];

                u32DesScoreIndex = SVP_WK_SCORE_NUM * u32DesBox;
                pf32Scores[u32DesScoreIndex + 0] = (HI_FLOAT)ps32Src[u32SrcBgProbIndex] / SVP_WK_QUANT_BASE;
                pf32Scores[u32DesScoreIndex + 1] = (HI_FLOAT)ps32Src[u32SrcFgProbIndex] / SVP_WK_QUANT_BASE;
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
    s32Ret = FilterLowScoreBbox( ps32Proposals, u32NumAnchors, u32NmsThresh, u32FilterThresh, &u32NumAfterFilter );
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

/**********************************************************************************************
Function    : RoIpooling
Description : Do Roipooling for each proposal generated by RPN
Input       :
***********************************************************************************************/
HI_S32 HI_MPI_SVP_NNIE_WK_CNN_ROIPOOLING_Ref(
    HI_S32* ps32Conv,
    HI_S32* ps32Proposals,
    HI_U32 u32RoiNum,
    NNIE_MODEL_INFO_S* pstModelInfo,
    HI_S32 *ps32AssistMem,
    HI_S32* ps32Roi)
{
    printf("ROIpooling function begins\n");
    /************** define parameters ***************/
    HI_U32 u32Conv1Height    = 0;
    HI_U32 u32Conv1Width     = 0;

    HI_U32 u32RoiStartW      = 0;
    HI_U32 u32RoiStartH      = 0;
    HI_U32 u32RoiEndW        = 0;
    HI_U32 u32RoiEndH        = 0;
    HI_U32 u32RoiHeight      = 0;
    HI_U32 u32RoiWidth       = 0;
    HI_U32 u32HStart         = 0;
    HI_U32 u32HEnd           = 0;
    HI_U32 u32WStart         = 0;
    HI_U32 u32WEnd           = 0;
    HI_U32 u32SrcIndex       = 0;
    HI_U32 u32DesIndex       = 0;
    HI_FLOAT f32BinSizeH     = 0;
    HI_FLOAT f32BinSizeW     = 0;
    HI_FLOAT f32SpatialScale = 0;
    HI_U32 u32ConvMapSize    = 0;
    HI_U32 u32ConvLineSize   = 0;
    HI_U32 u32RoiMapSize     = 0;
    HI_U32 u32RoiLineSize    = 0;
    HI_U32 u32PoolHeight     = 0;
    HI_U32 u32PoolWidth      = 0;
    HI_U32 u32RoiChannel     = 0;
    HI_U32 u32OutputRoiSize  = 0;

    HI_U32 u32OldIndex       = 0;
    HI_U32 u32NewIndex       = 0;
    HI_U32 u32PermMapSize    = 0;
    HI_U32 u32PermLineSize   = 0;
    HI_S32* pst32PermData    = NULL;
    HI_S32* ps32ChannelMax   = NULL;

    HI_U32 i  = 0;
    HI_U32 j  = 0;
    HI_U32 c  = 0;
    HI_U32 h  = 0;
    HI_U32 w  = 0;
    HI_U32 ph = 0;
    HI_U32 pw = 0;

    u32Conv1Height   = pstModelInfo->astReportNodeInfo[0].u32ConvHeight;
    u32Conv1Width    = pstModelInfo->astReportNodeInfo[0].u32ConvWidth;

    u32PoolHeight    = pstModelInfo->u32RoiHeight;
    u32PoolWidth     = pstModelInfo->u32RoiWidth;
    u32RoiChannel    = pstModelInfo->u32RoiMapNum;
    f32SpatialScale  = (HI_FLOAT)(pstModelInfo->u32SpatialScale) / SVP_WK_QUANT_BASE;

    u32ConvMapSize   = ALIGN32(u32Conv1Width * sizeof(HI_U32)) * u32Conv1Height / sizeof(HI_U32);
    u32ConvLineSize  = ALIGN32(u32Conv1Width * sizeof(HI_U32)) / sizeof(HI_U32);
    u32RoiMapSize    = u32PoolHeight * ALIGN32(u32PoolWidth);
    u32RoiLineSize   = ALIGN32(u32PoolWidth);

    u32OutputRoiSize = ALIGN32(pstModelInfo->u32RoiWidth) * pstModelInfo->u32RoiHeight * pstModelInfo->u32RoiMapNum;

    /************************ get the pst32PermData **********************/
    pst32PermData   = ps32AssistMem;
    u32PermMapSize  = u32Conv1Width * u32RoiChannel;
    u32PermLineSize = u32RoiChannel;

    ps32ChannelMax  = pst32PermData + u32Conv1Height * u32Conv1Width * u32RoiChannel;

    /*********************** permute ***********************/
    for (c = 0; c < u32RoiChannel; c++)
    {
        for (h = 0; h < u32Conv1Height; h++)
        {
            for (w = 0; w < u32Conv1Width; w++)
            {
                u32OldIndex = c * u32ConvMapSize + h * u32ConvLineSize + w;
                u32NewIndex = h * u32PermMapSize + w * u32PermLineSize + c;
                pst32PermData[u32NewIndex] = ps32Conv[u32OldIndex];
            }
        }
    }

    /***************** for each proposal ****************/
    for (i = 0; i < u32RoiNum; i++)
    {
        u32RoiStartW = (HI_U32)SAFE_ROUND(ps32Proposals[SVP_WK_COORDI_NUM * i] * f32SpatialScale);
        u32RoiStartH = (HI_U32)SAFE_ROUND(ps32Proposals[SVP_WK_COORDI_NUM * i + 1] * f32SpatialScale);
        u32RoiEndW   = (HI_U32)SAFE_ROUND(ps32Proposals[SVP_WK_COORDI_NUM * i + 2] * f32SpatialScale);
        u32RoiEndH   = (HI_U32)SAFE_ROUND(ps32Proposals[SVP_WK_COORDI_NUM * i + 3] * f32SpatialScale);

        u32RoiHeight = SVP_MAX(u32RoiEndH - u32RoiStartH + 1, 1);
        u32RoiWidth  = SVP_MAX(u32RoiEndW - u32RoiStartW + 1, 1);

        f32BinSizeH = (HI_FLOAT)u32RoiHeight / u32PoolHeight;
        f32BinSizeW = (HI_FLOAT)u32RoiWidth / u32PoolWidth;

        for (ph = 0; ph < u32PoolHeight; ph++)
        {
            u32HStart = (HI_U32)(floor(ph * f32BinSizeH));
            u32HEnd   = (HI_U32)(ceil((ph + 1)*f32BinSizeH));
            u32HStart = SVP_MIN(SVP_MAX(u32HStart + u32RoiStartH, 0), u32Conv1Height);
            u32HEnd   = SVP_MIN(SVP_MAX(u32HEnd + u32RoiStartH, 0), u32Conv1Height);

            for (pw = 0; pw < u32PoolWidth; pw++)
            {
                u32WStart = (HI_U32)(floor(pw * f32BinSizeW));
                u32WEnd   = (HI_U32)(ceil((pw + 1)*f32BinSizeW));
                u32WStart = SVP_MIN(SVP_MAX(u32WStart + u32RoiStartW, 0), u32Conv1Width);
                u32WEnd   = SVP_MIN(SVP_MAX(u32WEnd + u32RoiStartW, 0), u32Conv1Width);

                for (c = 0; c < u32RoiChannel; c++)
                {
                    ps32ChannelMax[c] = 0;
                }

                for (h = u32HStart; h < u32HEnd; h++)
                {
                    for (w = u32WStart; w < u32WEnd; w++)
                    {
                        u32SrcIndex = h * u32PermMapSize + w * u32PermLineSize;
                        for (j = 0; j < u32RoiChannel; j++)
                        {
                            if (pst32PermData[u32SrcIndex] > ps32ChannelMax[j])
                            {
                                ps32ChannelMax[j] = pst32PermData[u32SrcIndex];
                            }
                            u32SrcIndex++;
                        }
                    }
                }

                for (c = 0; c < u32RoiChannel; c++)
                {
                    u32DesIndex = c * u32RoiMapSize + ph * u32RoiLineSize + pw;
                    ps32Roi[u32DesIndex] = ps32ChannelMax[c];
                }

            }
        }
        ps32Roi += u32OutputRoiSize;
    }

    printf("ROIpooling function complete\n");
    return HI_SUCCESS;
}

HI_S32  HI_MPI_SVP_NNIE_WK_RFCN_GetResult(SVP_SRC_MEM_INFO_S *pstSrc, HI_U32 u32FcScoreStride, HI_S32* ps32FcBbox, HI_U32 u32FcBboxStride,
    SVP_SRC_MEM_INFO_S *pstProposal, HI_U32 u32RoiCnt,
    SVP_DST_MEM_INFO_S *pstDstScore, SVP_DST_MEM_INFO_S *pstDstBbox,
    SVP_MEM_INFO_S *pstRoiOutCnt, NNIE_MODEL_INFO_S *pstModelInfo,
    HI_U32 u32ConfThresh, HI_U32 u32NmsThresh, SVP_MEM_INFO_S *pstMemPool, std::vector<RFCN_BoxesInfo> &vBoxesInfo,
    std::string& resultPath, SVP_SAMPLE_FILE_NAME_PAIR& imgNamePair)
{
    /************* define variables *****************/
    HI_U32 u32Size                 = 0;
    HI_U32 u32ClsScoreChannels     = 0;
    HI_S32* ps32FcScore            = NULL;
    HI_S32* ps32Proposals          = NULL;
    HI_U32 u32FcScoreWidth         = 0;
    HI_U32 u32FcBboxWidth          = 0;
    HI_U32 u32OriImWidth           = 0;
    HI_U32 u32OriImHeight          = 0;

    HI_U32 u32MaxRoi               = 0;
    HI_FLOAT* pf32FcScoresMemPool  = NULL;
    HI_S32* ps32FcBboxMemPool      = NULL;
    HI_S32* ps32ProposalMemPool    = NULL;
    HI_S32* ps32ProposalTmp        = NULL;
    HI_U32 u32ProposalMemPoolIndex = 0;
    HI_FLOAT* pf32Ptr              = NULL;
    HI_S32* ps32Ptr                = NULL;
    HI_S32* ps32DstScore           = NULL;
    HI_S32* ps32DstBbox            = NULL;
    HI_S32* ps32RoiOutCnt          = NULL;
    HI_U32 u32RoiOutCnt            = 0;
    NNIE_STACK_S* pstStack         = NULL;
    HI_S32 s32Ret                  = HI_FAILURE;
    HI_FLOAT f32ScoreOut           = 0.0;
    HI_S32 s32XMin                 = 0;
    HI_S32 s32YMin                 = 0;
    HI_S32 s32XMax                 = 0;
    HI_S32 s32YMax                 = 0;
    HI_U32 u32SrcIndex             = 0;
    HI_U32 u32DstIndex             = 0;

    HI_U32 i = 0;
    HI_U32 j = 0;

    if (SVP_NNIE_NET_TYPE_ROI == pstModelInfo->enNetType)
    {
        u32MaxRoi = pstModelInfo->u32MaxRoiFrameCnt;
        /********* check the proposal size *********/

        /******************* Get or calculate parameters **********************/
        u32ClsScoreChannels = pstModelInfo->u32ClassSize;   /*channel num is equal to class size, cls_score class*/

        u32Size = u32RoiCnt*u32ClsScoreChannels;              /*s32Height*s32Width;*/

        u32FcScoreWidth = u32FcScoreStride / sizeof(HI_U32);
        u32FcBboxWidth = u32FcBboxStride / sizeof(HI_U32);

        ps32FcScore = (HI_S32*)pstSrc->u64VirAddr;  /* Pointer to scores (ave pooling) */
        u32Size = u32MaxRoi * u32FcScoreWidth;
        u32OriImWidth = pstModelInfo->u32SrcWidth;
        u32OriImHeight = pstModelInfo->u32SrcHeight;

        /*************** Get Start Pointer of MemPool ******************/
        pf32FcScoresMemPool = (HI_FLOAT*)(pstMemPool->u64VirAddr);
        pf32Ptr = pf32FcScoresMemPool;
        u32Size = u32MaxRoi * u32ClsScoreChannels;
        pf32Ptr += u32Size;

        ps32FcBboxMemPool = (HI_S32*)pf32Ptr;
        ps32Ptr = (HI_S32*)pf32Ptr;
        u32Size = u32MaxRoi * SVP_WK_COORDI_NUM;
        ps32Ptr += u32Size;

        ps32ProposalMemPool = (HI_S32*)ps32Ptr;
        ps32Ptr = ps32ProposalMemPool;
        u32Size = u32MaxRoi * SVP_WK_PROPOSAL_WIDTH;
        ps32Ptr += u32Size;
        pstStack = (struct hiNNIE_STACK*)ps32Ptr;

        // prepare input data
        for (i = 0; i < u32RoiCnt; i++)
        {
            for (j = 0; j < u32ClsScoreChannels; j++)
            {
                u32SrcIndex = u32FcScoreWidth * i + j;
                pf32FcScoresMemPool[u32DstIndex++] = (HI_FLOAT)ps32FcScore[u32SrcIndex] / SVP_WK_QUANT_BASE;
            }
        }

        u32DstIndex = 0;
        for (i = 0; i < u32RoiCnt; i++)
        {
            for (j = 0; j < SVP_WK_COORDI_NUM; j++)
            {
                u32SrcIndex = u32FcBboxWidth * i + SVP_WK_COORDI_NUM + j;
                u32DstIndex = SVP_WK_COORDI_NUM * i + j;
                ps32FcBboxMemPool[u32DstIndex] = ps32FcBbox[u32SrcIndex];
            }
        }


#if 0
        FILE* fc_bbox = fopen((resultPath + imgNamePair.first + "_fc_bbox.txt").c_str(), "w");
        SVP_CHECK(NULL != fc_bbox, HI_FAILURE);

        for (i = 0; i < u32RoiCnt; i++)
        {
            for (j = 0; j < SVP_WK_COORDI_NUM; j++)
            {
                fprintf(fc_bbox, "%d\n", ps32FcBbox[i*SVP_WK_COORDI_NUM + j]);
            }
        }
        fclose(fc_bbox);
        fc_bbox = NULL;
#endif

        ps32Proposals = (HI_S32*)pstProposal->u64VirAddr;

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
        string fileName = resultPath + imgNamePair.first + "_"+ pre_tmp + "ALL" + post_tmp;
        ofstream foutALL(fileName.c_str());
        SVP_CHECK(foutALL.good(), HI_FAILURE);

        HI_BOOL hasResultOut = HI_FALSE;

        PrintBreakLine(HI_TRUE);

        ss << u32OriImWidth << "  " << u32OriImHeight << endl;
        cout << ss.str();
        foutALL << ss.str();
        ss.str("");

        HI_FLOAT pf32Proposals[SVP_WK_PROPOSAL_WIDTH];
        HI_FLOAT pf32Anchors[SVP_WK_COORDI_NUM];
        HI_FLOAT pf32BboxDelta[SVP_WK_COORDI_NUM];
        for (j = 0; j < u32ClsScoreChannels; j++)
        {
            for (i = 0; i < u32RoiCnt; i++)
            {
                for (HI_U32 k = 0; k < SVP_WK_COORDI_NUM; k++)
                {
                    pf32Anchors[k] = (HI_FLOAT)ps32Proposals[SVP_WK_COORDI_NUM*i + k] / SVP_WK_QUANT_BASE;
                    pf32BboxDelta[k] = (HI_FLOAT)ps32FcBboxMemPool[SVP_WK_COORDI_NUM*i + k];
                }

                s32Ret = BboxTransform_FLOAT(
                    pf32Proposals,
                    pf32Anchors,
                    pf32BboxDelta,
                    &pf32FcScoresMemPool[u32ClsScoreChannels * i + j]);

                SVP_CHECK(HI_SUCCESS == s32Ret, HI_FAILURE);

                for (HI_U32 k = 0; k < SVP_WK_PROPOSAL_WIDTH; k++)
                {
                    ps32ProposalMemPool[i*SVP_WK_PROPOSAL_WIDTH + k] = (HI_S32)pf32Proposals[k];
                }

            }

            /* clip bbox */
            s32Ret = BboxClip_N(ps32ProposalMemPool, u32OriImWidth, u32OriImHeight, u32RoiCnt);
            SVP_CHECK(HI_SUCCESS == s32Ret, HI_FAILURE);

            ps32ProposalTmp = ps32ProposalMemPool;

            /* sort */
            s32Ret = NonRecursiveArgQuickSort(ps32ProposalTmp, 0, (HI_S32)u32RoiCnt - 1, pstStack);
            SVP_CHECK(HI_SUCCESS == s32Ret, HI_FAILURE);

            /* NMS */
            s32Ret = NonMaxSuppression(ps32ProposalTmp, u32RoiCnt, u32NmsThresh);
            SVP_CHECK(HI_SUCCESS == s32Ret, HI_FAILURE);

            pre_tmp = "roi_pos_out_";
            post_tmp = ".txt";
            fileName = resultPath + imgNamePair.first + "_" + pre_tmp + to_string((HI_U64)j) + post_tmp;
            ofstream fout(fileName.c_str());
            SVP_CHECK(fout.good(), HI_FAILURE);

            u32RoiOutCnt  = 0;
            ps32DstScore  = (HI_S32*)pstDstScore->u64VirAddr;
            ps32DstBbox   = (HI_S32*)pstDstBbox->u64VirAddr;
            ps32RoiOutCnt = (HI_S32*)pstRoiOutCnt->u64VirAddr;

            ps32DstScore += (HI_S32)(j * u32MaxRoi);
            ps32DstBbox  += (HI_S32)(j * SVP_WK_COORDI_NUM * u32MaxRoi);

            for (i = 0; i < u32RoiCnt; i++)
            {
                u32ProposalMemPoolIndex = SVP_WK_PROPOSAL_WIDTH * i;
                if (RPN_SUPPRESS_FALSE == ps32ProposalMemPool[u32ProposalMemPoolIndex + 5] &&
                    ps32ProposalMemPool[u32ProposalMemPoolIndex + 4] > (HI_S32)u32ConfThresh)
                {
                    ps32DstScore[u32RoiOutCnt] = ps32ProposalMemPool[u32ProposalMemPoolIndex + 4];
                    ps32DstBbox[u32RoiOutCnt * SVP_WK_COORDI_NUM + 0] = ps32ProposalMemPool[u32ProposalMemPoolIndex] + 0;
                    ps32DstBbox[u32RoiOutCnt * SVP_WK_COORDI_NUM + 1] = ps32ProposalMemPool[u32ProposalMemPoolIndex + 1];
                    ps32DstBbox[u32RoiOutCnt * SVP_WK_COORDI_NUM + 2] = ps32ProposalMemPool[u32ProposalMemPoolIndex + 2];
                    ps32DstBbox[u32RoiOutCnt * SVP_WK_COORDI_NUM + 3] = ps32ProposalMemPool[u32ProposalMemPoolIndex + 3];

                    f32ScoreOut = (HI_FLOAT)ps32DstScore[u32RoiOutCnt] / SVP_WK_QUANT_BASE;
                    if (f32ScoreOut > 1.0 || f32ScoreOut < 0.0)
                    {
                        printf("ERROR in f32ScoreOut(%f), out of range[0,1]\n", f32ScoreOut);
                    }
                    else
                    {
                        s32XMin = ps32DstBbox[u32RoiOutCnt * SVP_WK_COORDI_NUM + 0];
                        s32YMin = ps32DstBbox[u32RoiOutCnt * SVP_WK_COORDI_NUM + 1];
                        s32XMax = ps32DstBbox[u32RoiOutCnt * SVP_WK_COORDI_NUM + 2];
                        s32YMax = ps32DstBbox[u32RoiOutCnt * SVP_WK_COORDI_NUM + 3];

                        ss << imgNamePair.first << "  " <<
                            setw(4) << j << "  " <<
                            fixed << setprecision(8) << f32ScoreOut << "  " <<
                            setw(4) << s32XMin << "  " <<
                            setw(4) << s32YMin << "  " <<
                            setw(4) << s32XMax << "  " <<
                            setw(4) << s32YMax << endl;


                        hasResultOut = HI_TRUE;
                        /* dump to roi_pos_out_ALL.txt except class 0*/
                        if (0 != j) {
                            foutALL << ss.str();
                            RFCN_BoxesInfo stBoxInfo = { (HI_U32)s32XMin, (HI_U32)s32YMin, (HI_U32)s32XMax, (HI_U32)s32YMax, j, f32ScoreOut };
                            vBoxesInfo.push_back(stBoxInfo);
                        }

                        /* dump to roi_pos_out_i.txt */
                        fout << ss.str();
                        cout << ss.str();
                        ss.str("");
                   }

                    u32RoiOutCnt++;
                }
                if (u32RoiOutCnt >= u32RoiCnt)break;
            }

            ps32RoiOutCnt[j] = (HI_S32)u32RoiOutCnt;

            fout.close();

            PrintBreakLine(hasResultOut);
            hasResultOut = HI_FALSE;
        }
        foutALL.close();

    }
    return HI_SUCCESS;
}
