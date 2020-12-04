
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <string>
#include <iomanip>

#include "fasterrcnn_interface.h"
#include "detectionCom.h"

using namespace std;

HI_S32 HI_MPI_SVP_NNIE_WK_CNN_FASTER_RPN_Ref1(
    HI_S32** ps32Src,
    HI_U32 u32NumRatioAnchors,
    HI_U32 u32NumScaleAnchors,
    HI_U32* pu32Scales,
    HI_U32* pu32Ratios,
    HI_U32 u32OriImHeight,
    HI_U32 u32OriImWidth,
    HI_U32* pu32ConvHeight,
    HI_U32* pu32ConvWidth,
    HI_U32* pu32ConvChannel,
    HI_U32* pu32ConvStride,
    HI_U32 u32MaxRois,
    HI_U32 u32MinSize,
    HI_U32 u32SpatialScale,
    HI_U32 u32NmsThresh,
    HI_U32 u32FilterThresh,
    HI_U32 u32NumBeforeNms,
    HI_U32* pu32MemPool,
    HI_S32* ps32ProposalResult,
    HI_U32* pu32NumRois)
{
    printf("RPN function begins\n");

    /******************** define parameters ****************/
    HI_U32 u32Size              = 0;
    HI_U32* pu32Ptr             = NULL;

    HI_FLOAT* pf32Ptr           = NULL;
    HI_S32* ps32Anchors         = NULL;
    HI_S32* ps32BboxDelta       = NULL;
    HI_S32* ps32Proposals       = NULL;
    HI_FLOAT* pf32RatioAnchors  = NULL;
    HI_FLOAT* pf32ScaleAnchors  = NULL;
    HI_FLOAT* pf32Scores        = NULL;

    HI_U32 u32Conv2Height       = 0;
    HI_U32 u32Conv2Width        = 0;
    HI_U32 u32Conv3Height       = 0;
    HI_U32 u32Conv3Width        = 0;
    HI_U32 u32Conv3Channel      = 0;
    HI_U32 u32Conv3Stride       = 0;
    HI_U32 u32NumAfterFilter    = 0;
    HI_U32 u32NumAnchors        = 0;

    HI_U32 u32SrcBboxIndex      = 0;
    HI_U32 u32SrcFgProbIndex    = 0;
    HI_U32 u32SrcBgProbIndex    = 0;
    HI_U32 u32SrcBboxBias       = 0;
    HI_U32 u32SrcProbBias       = 0;
    HI_U32 u32DesBox            = 0;
    HI_U32 u32BgBlobSize        = 0;
    HI_U32 u32AnchorsPerPixel   = 0;
    HI_U32 u32MapSize           = 0;
    HI_U32 u32LineSize          = 0;
    HI_U32 u32DesBboxDeltaIndex = 0;
    HI_U32 u32DesScoreIndex     = 0;
    NNIE_STACK_S* pstStack      = NULL;
    HI_S32 s32Ret               = HI_FAILURE;

    u32Conv2Height  = pu32ConvHeight[1];
    u32Conv2Width   = pu32ConvWidth[1];

    u32Conv3Height  = pu32ConvHeight[2];
    u32Conv3Width   = pu32ConvWidth[2];
    u32Conv3Channel = pu32ConvChannel[2];
    u32Conv3Stride  = pu32ConvStride[2];

    HI_U32 au32BaseAnchor[SVP_WK_COORDI_NUM] = { 0, 0, (u32MinSize - 1), (u32MinSize - 1) };

    /*********************************** Faster RCNN *********************************************/
    /********* calculate the start pointer of each part in MemPool *********/
    pu32Ptr = (HI_U32*)pu32MemPool;
    ps32Anchors = (HI_S32*)pu32Ptr;
    u32NumAnchors = u32NumRatioAnchors * u32NumScaleAnchors * (u32Conv2Height * u32Conv2Width);
    u32Size = SVP_WK_COORDI_NUM * u32NumAnchors;
    pu32Ptr += u32Size;

    ps32BboxDelta = (HI_S32*)pu32Ptr;
    pu32Ptr += u32Size;

    ps32Proposals = (HI_S32*)pu32Ptr;
    u32Size = SVP_WK_PROPOSAL_WIDTH * u32NumAnchors;
    pu32Ptr += u32Size;

    pf32RatioAnchors = (HI_FLOAT*)pu32Ptr;
    pf32Ptr = (HI_FLOAT*)pu32Ptr;
    u32Size = u32NumRatioAnchors * SVP_WK_COORDI_NUM;
    pf32Ptr = pf32Ptr + u32Size;

    pf32ScaleAnchors = pf32Ptr;
    u32Size = u32NumScaleAnchors * u32NumRatioAnchors * SVP_WK_COORDI_NUM;
    pf32Ptr = pf32Ptr + u32Size;

    pf32Scores = pf32Ptr;
    u32Size = u32NumAnchors * (SVP_WK_SCORE_NUM);
    pf32Ptr = pf32Ptr + u32Size;

    pstStack = (NNIE_STACK_S*)pf32Ptr;

    /********************* Generate the base anchor ***********************/
    s32Ret = GenBaseAnchor(pf32RatioAnchors, pu32Ratios, u32NumRatioAnchors,
                           pf32ScaleAnchors, pu32Scales, u32NumScaleAnchors,
                           au32BaseAnchor);
    SVP_CHECK(HI_SUCCESS == s32Ret, HI_FAILURE);

    /******************* Copy the anchors to every pixel in the feature map ******************/
    s32Ret = SetAnchorInPixel(ps32Anchors,
                              pf32ScaleAnchors,
                              u32Conv2Height,
                              u32Conv2Width,
                              u32NumScaleAnchors * u32NumRatioAnchors,
                              u32SpatialScale);
    SVP_CHECK(HI_SUCCESS == s32Ret, HI_FAILURE);

    /********** do transpose, convert the blob from (M,C,H,W) to (M,H,W,C) **********/
    u32MapSize = u32Conv3Height * (u32Conv3Stride / sizeof(HI_U32));
    u32AnchorsPerPixel = u32NumRatioAnchors * u32NumScaleAnchors;
    u32BgBlobSize = u32AnchorsPerPixel * u32MapSize;
    u32LineSize = u32Conv3Stride / sizeof(HI_U32);
    u32SrcProbBias = 0;
    u32SrcBboxBias = 0;

    for (HI_U32 c = 0; c < u32Conv3Channel; c++)
    {
        for (HI_U32 h = 0; h < u32Conv3Height; h++)
        {
            for (HI_U32 w = 0; w < u32Conv3Width; w++)
            {
                u32SrcBboxIndex = u32SrcBboxBias + c * u32MapSize + h * u32LineSize + w;
                u32SrcBgProbIndex = u32SrcProbBias + (c / SVP_WK_COORDI_NUM) * u32MapSize + h * u32LineSize + w;
                u32SrcFgProbIndex = u32BgBlobSize + u32SrcBgProbIndex;

                u32DesBox = u32AnchorsPerPixel * (h * u32Conv3Width + w) + c / SVP_WK_COORDI_NUM;

                u32DesBboxDeltaIndex = u32DesBox * SVP_WK_COORDI_NUM  + c % SVP_WK_COORDI_NUM;
                ps32BboxDelta[u32DesBboxDeltaIndex] = (HI_S32)ps32Src[2][u32SrcBboxIndex];

                u32DesScoreIndex = (SVP_WK_SCORE_NUM)* u32DesBox;
                pf32Scores[u32DesScoreIndex + 0] = (HI_FLOAT)(ps32Src[1][u32SrcBgProbIndex]) / SVP_WK_QUANT_BASE;
                pf32Scores[u32DesScoreIndex + 1] = (HI_FLOAT)(ps32Src[1][u32SrcFgProbIndex]) / SVP_WK_QUANT_BASE;
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
    s32Ret = BboxSmallSizeFilter_N(ps32Proposals, u32MinSize, u32MinSize, u32NumAnchors);
    SVP_CHECK(HI_SUCCESS == s32Ret, HI_FAILURE);

    /********** remove low score bboxes ************/
    s32Ret = FilterLowScoreBbox(ps32Proposals, u32NumAnchors, u32NmsThresh, u32FilterThresh, &u32NumAfterFilter);
    SVP_CHECK(HI_SUCCESS == s32Ret, HI_FAILURE);

    /************* sort with NonRecursiveArgQuickSort **************/
    s32Ret = NonRecursiveArgQuickSort(ps32Proposals, 0, (HI_S32)u32NumAfterFilter - 1, pstStack);
    SVP_CHECK(HI_SUCCESS == s32Ret, HI_FAILURE);

    u32NumAfterFilter = SVP_MIN(u32NumAfterFilter, u32NumBeforeNms);

    /****************** write Proposal_before_nms_rpn *********************/
    s32Ret = dumpProposal(ps32Proposals, "Proposal_before_nms_rpn.txt", u32NumAnchors);
    SVP_CHECK(HI_SUCCESS == s32Ret, HI_FAILURE);

    /****************** do nms to remove highly overlapped bbox **********/
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

}

 HI_S32  HI_MPI_SVP_NNIE_WK_RCNN_GetResult1(
    HI_S32** pstSrc,
    HI_U32  bbox_stride,
    HI_U32  score_stride,
    HI_S32* pstProposal,
    HI_U32  u32RoiCnt,
    HI_U32  u32ConfThresh,
    HI_S32* pstDstScore,
    HI_S32* pstDstBbox,
    HI_S32* pstRoiOutCnt,
    HI_U32 u32NmsThresh,
    HI_U32 u32MaxRoi,
    HI_U32 u32ClassNum,
    HI_U32 u32OriImWidth,
    HI_U32 u32OriImHeight,
    HI_S32* pstMemPool)  // assist mem
{
    /************* define variables *****************/

    HI_U32 u32Size                 = 0;
    HI_U32 u32ClsScoreChannels     = 0;

    HI_S32* ps32FcScore            = NULL;
    HI_S32* ps32FcBbox             = NULL;
    HI_S32* ps32Proposals          = NULL;
    HI_U32 u32FcScoreWidth         = 0;
    HI_U32 u32FcBboxWidth          = 0;

    HI_FLOAT* pf32FcScoresMemPool  = NULL;
    HI_S32* ps32ProposalMemPool    = NULL;
    HI_S32* ps32ProposalTmp        = NULL;

    HI_U32 u32ProposalMemPoolIndex = 0;
    HI_FLOAT* pf32Ptr              = NULL;
    HI_S32* ps32Ptr                = NULL;
    HI_S32* ps32DstScore           = NULL;
    HI_S32* ps32DstBbox            = NULL;
    HI_S32* ps32RoiOutCnt          = NULL;
    HI_U32 u32RoiOutCnt            = 0;

    HI_U32 u32SrcIndex             = 0;
    HI_U32 u32DstIndex             = 0;

    HI_U32 i = 0;
    HI_U32 j = 0;
    HI_U32 k = 0;

    NNIE_STACK_S* pstStack = NULL;
    HI_S32 s32Ret = HI_FAILURE;

    /******************* Get or calculate parameters **********************/
    u32ClsScoreChannels = u32ClassNum;                         /*channel num is equal to class size, cls_score class*/
    u32Size = u32RoiCnt * u32ClsScoreChannels;                 /*s32Height*s32Width;*/
    u32FcScoreWidth = score_stride / sizeof(HI_U32);
    u32FcBboxWidth = bbox_stride / sizeof(HI_U32);
    ps32FcScore = (HI_S32*)pstSrc[0];                          /* Pointer to FC scores */
    u32Size = u32MaxRoi * u32FcBboxWidth;
    ps32FcBbox = (HI_S32*)pstSrc[1];                           /* Pointer to FC Bbox */

    /*************** Get Start Pointer of MemPool ******************/
    pf32FcScoresMemPool = (HI_FLOAT*)pstMemPool;
    pf32Ptr = pf32FcScoresMemPool;
    u32Size = u32MaxRoi * u32ClsScoreChannels;
    pf32Ptr += u32Size;

    ps32ProposalMemPool = (HI_S32*)pf32Ptr;
    ps32Ptr = ps32ProposalMemPool;
    u32Size = u32MaxRoi * SVP_WK_PROPOSAL_WIDTH;
    ps32Ptr += u32Size;
    pstStack = (struct hiNNIE_STACK*)ps32Ptr;

    u32DstIndex = 0;

    for (i = 0; i < u32RoiCnt; i++)
    {
        for (k = 0; k < u32ClsScoreChannels; k++)
        {
            u32SrcIndex = i * u32FcScoreWidth + k;
            pf32FcScoresMemPool[u32DstIndex++] = (HI_FLOAT)ps32FcScore[u32SrcIndex] / SVP_WK_QUANT_BASE;
        }
    }

    /****************** write fc_bbox *********************/
#if 0
    FILE* fc_bbox = fopen("fc_bbox.txt", "w");
    SVP_CHECK(NULL != fc_bbox, HI_FAILURE);

    printf("The Size of Fc Score is: %d\n", u32MaxRoi * u32FcScoreWidth);
    for (i = 0; i < u32RoiCnt; i++)
    {
        for (j = 0; j < u32FcBboxWidth; j++)
        {
            fprintf(fc_bbox, "%d\n", ps32FcBbox[i*u32FcBboxWidth + j]);
        }
    }
    fclose(fc_bbox);
    fc_bbox = NULL;
#endif

    ps32Proposals = (HI_S32*)pstProposal;

    /************** bbox transform ************
    change the fc output to Proposal temp MemPool.
    Each Line of the Proposal has 6 bits.
    The Format of the Proposal is:
    0-3: The four coordinate of the bbox, x1,y1,x2,y2
    4: The Confidence Score of the bbox
    5: The suppressed flag
    ******************************************/

    HI_FLOAT pf32Proposals[SVP_WK_PROPOSAL_WIDTH];
    HI_FLOAT pf32Anchors[SVP_WK_COORDI_NUM];
    HI_FLOAT pf32BboxDelta[SVP_WK_COORDI_NUM];
    for(j = 0; j < u32ClsScoreChannels; j++)
    {
        for (i = 0; i < u32RoiCnt; i++)
        {
            for (k = 0; k < SVP_WK_COORDI_NUM; k++)
            {
                pf32Anchors[k] = (HI_FLOAT)ps32Proposals[SVP_WK_COORDI_NUM*i+k] / SVP_WK_QUANT_BASE;
                pf32BboxDelta[k] = (HI_FLOAT)ps32FcBbox[u32FcBboxWidth*i + SVP_WK_COORDI_NUM * j + k];
            }

            s32Ret = BboxTransform_FLOAT(pf32Proposals,
                                         pf32Anchors,
                                         pf32BboxDelta,
                                         &pf32FcScoresMemPool[u32ClsScoreChannels*i + j]);

            SVP_CHECK(HI_SUCCESS == s32Ret, HI_FAILURE);

            for (k = 0; k < SVP_WK_PROPOSAL_WIDTH; k++)
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

        u32RoiOutCnt = 0;
        ps32DstScore  = pstDstScore;
        ps32DstBbox   = pstDstBbox;
        ps32RoiOutCnt = pstRoiOutCnt;
        ps32DstScore += (HI_S32)(j * u32MaxRoi);
        ps32DstBbox  += (HI_S32)(j * u32MaxRoi * SVP_WK_COORDI_NUM);

        for(i = 0; i < u32RoiCnt; i++)
        {
            u32ProposalMemPoolIndex = SVP_WK_PROPOSAL_WIDTH * i;
            if( RPN_SUPPRESS_FALSE == ps32ProposalMemPool[u32ProposalMemPoolIndex + 5] &&
                ps32ProposalMemPool[u32ProposalMemPoolIndex + 4] > (HI_S32)u32ConfThresh )
            {
                ps32DstScore[u32RoiOutCnt] = ps32ProposalMemPool[u32ProposalMemPoolIndex + 4];
                ps32DstBbox[u32RoiOutCnt * SVP_WK_COORDI_NUM + 0 ] = ps32ProposalMemPool[u32ProposalMemPoolIndex + 0];
                ps32DstBbox[u32RoiOutCnt * SVP_WK_COORDI_NUM + 1 ] = ps32ProposalMemPool[u32ProposalMemPoolIndex + 1];
                ps32DstBbox[u32RoiOutCnt * SVP_WK_COORDI_NUM + 2 ] = ps32ProposalMemPool[u32ProposalMemPoolIndex + 2];
                ps32DstBbox[u32RoiOutCnt * SVP_WK_COORDI_NUM + 3 ] = ps32ProposalMemPool[u32ProposalMemPoolIndex + 3];

                u32RoiOutCnt++;
            }
            if(u32RoiOutCnt >= u32RoiCnt)break;
        }
        ps32RoiOutCnt[j] = (HI_S32)u32RoiOutCnt;
    }

    return HI_SUCCESS;
}

HI_S32 HI_MPI_SVP_NNIE_WK_CNN_ROIPOOLING_Ref(
    HI_S32* ps32Conv,
    HI_S32* ps32Proposals,
    HI_U32 u32RoiNum,
    HI_U32 u32SpatialScale,
    HI_S32* ps32Roi,
    HI_U32 u32Conv1Height,
    HI_U32 u32Conv1Width,
    HI_U32 u32PoolHeight,
    HI_U32 u32PoolWidth,
    HI_U32 u32RoiChannel)
{
    printf("ROIpooling function begins\n");

    /************** define parameters ***************/
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

    HI_U32 u32OutputRoiSize  = 0;
    HI_U32 i  = 0;
    HI_U32 j  = 0;
    HI_U32 h  = 0;
    HI_U32 w  = 0;
    HI_U32 ph = 0;
    HI_U32 pw = 0;

    /*************** check the input parameters **************/
    f32SpatialScale = (HI_FLOAT)(u32SpatialScale) / SVP_WK_QUANT_BASE;

    u32ConvMapSize  = ALIGN16(u32Conv1Width * sizeof(HI_U32)) * u32Conv1Height / sizeof(HI_U32);
    u32ConvLineSize = ALIGN16(u32Conv1Width * sizeof(HI_U32)) / sizeof(HI_U32);
    u32RoiLineSize  = ALIGN16(u32PoolWidth  * sizeof(HI_U32)) / sizeof(HI_U32);
    u32RoiMapSize   = u32PoolHeight * u32RoiLineSize;

    u32OutputRoiSize = u32RoiMapSize * u32RoiChannel;

    //printf("------ ROI num : %d\n", u32RoiNum);

    /***************** for each proposal ****************/
    for (i = 0; i < u32RoiNum; i++)
    {
        u32RoiStartW = (HI_U32)SAFE_ROUND(ps32Proposals[SVP_WK_COORDI_NUM * i] * f32SpatialScale);
        u32RoiStartH = (HI_U32)SAFE_ROUND(ps32Proposals[SVP_WK_COORDI_NUM * i + 1] * f32SpatialScale);
        u32RoiEndW   = (HI_U32)SAFE_ROUND(ps32Proposals[SVP_WK_COORDI_NUM * i + 2] * f32SpatialScale);
        u32RoiEndH   = (HI_U32)SAFE_ROUND(ps32Proposals[SVP_WK_COORDI_NUM * i + 3] * f32SpatialScale);

        u32RoiHeight = SVP_MAX(u32RoiEndH - u32RoiStartH + 1, 1);
        u32RoiWidth  = SVP_MAX(u32RoiEndW - u32RoiStartW + 1, 1);

        f32BinSizeH  = (HI_FLOAT)u32RoiHeight / u32PoolHeight;
        f32BinSizeW  = (HI_FLOAT)u32RoiWidth / u32PoolWidth;

        for (ph = 0; ph < u32PoolHeight; ph++)
        {
            for (pw = 0; pw < u32PoolWidth; pw++)
            {
                u32HStart = (HI_U32)(floor(ph * f32BinSizeH));
                u32WStart = (HI_U32)(floor(pw * f32BinSizeW));
                u32HEnd   = (HI_U32)(ceil((ph + 1)*f32BinSizeH));
                u32WEnd   = (HI_U32)(ceil((pw + 1)*f32BinSizeW));

                u32HStart = SVP_MIN(SVP_MAX(u32HStart + u32RoiStartH, 0), u32Conv1Height);
                u32HEnd   = SVP_MIN(SVP_MAX(u32HEnd + u32RoiStartH, 0), u32Conv1Height);
                u32WStart = SVP_MIN(SVP_MAX(u32WStart + u32RoiStartW, 0), u32Conv1Width);
                u32WEnd   = SVP_MIN(SVP_MAX(u32WEnd + u32RoiStartW, 0), u32Conv1Width);

                for (j = 0; j < u32RoiChannel; j++)    /* for each channel */
                {
                    u32DesIndex = j * u32RoiMapSize + ph * u32RoiLineSize + pw;
                    ps32Roi[u32DesIndex] = 0; /* since the input is from RELU layer, the mininum number is 0 */

                    for (h = u32HStart; h < u32HEnd; h++)   /* do max pooling in each bin */
                    {
                        for (w = u32WStart; w < u32WEnd; w++)
                        {
                            u32SrcIndex = j * u32ConvMapSize + h * u32ConvLineSize + w;
                            if (ps32Conv[u32SrcIndex] > ps32Roi[u32DesIndex])
                            {
                                ps32Roi[u32DesIndex] = ps32Conv[u32SrcIndex];
                            }
                        }
                    }
                }
            }
        }

        ps32Roi += u32OutputRoiSize;
    }

    printf("ROIpooling function complete\n");
    return HI_SUCCESS;
}

HI_S32 get_result_software(
    Faster_RCNN_Para* para,
    HI_S32** input_conv,
    HI_U32  bbox_stride,
    HI_U32  score_stride,
    HI_S32* rois,
    HI_S32* dst_score,
    HI_S32* dst_bbox,
    HI_S32* dst_roicnt,
    HI_S32* assist_mem
)
{
    HI_MPI_SVP_NNIE_WK_RCNN_GetResult1(
        input_conv,
        bbox_stride,
        score_stride,
        rois,
        para->u32NumRois,
        para->u32ConfThresh,
        dst_score,
        dst_bbox,
        dst_roicnt,
        para->u32ValidNmsThresh,
        para->u32MaxRois,
        para->u32ClassNum,
        para->u32OriImWidth,
        para->u32OriImHeight,
        assist_mem);  // assist mem

    return HI_SUCCESS;
}

HI_U32 malloc_rpn_assist_mem_size(Faster_RCNN_Para* para)
{
    HI_U32 num_anchors = (para->u32NumRatioAnchors) * (para->u32NumScaleAnchors) *
                         (para->au32ConvHeight[0]) * (para->au32ConvWidth[0]);

    HI_U32 size_anchor        = num_anchors * SVP_WK_COORDI_NUM * sizeof(HI_U32);
    HI_U32 size_bbox_delta    = size_anchor;
    HI_U32 size_proposal      = num_anchors * SVP_WK_PROPOSAL_WIDTH * sizeof(HI_U32);
    HI_U32 size_ratio_anchors = para->u32NumRatioAnchors * SVP_WK_COORDI_NUM * sizeof(HI_FLOAT);
    HI_U32 size_scale_anchors = para->u32NumRatioAnchors * para->u32NumScaleAnchors * SVP_WK_COORDI_NUM * sizeof(HI_FLOAT);
    HI_U32 size_scores        = num_anchors * SVP_WK_SCORE_NUM * sizeof(HI_FLOAT);
    HI_U32 size_stack         = MAX_STACK_DEPTH * sizeof(NNIE_STACK_S);

    HI_U32 total_size = size_anchor + size_bbox_delta +
                        size_proposal + size_ratio_anchors +
                        size_scale_anchors + size_scores + size_stack;

    return total_size;
}

HI_U32 malloc_get_result_assist_mem_size(Faster_RCNN_Para* para)
{
    HI_U32 size_scores   = para->u32MaxRois * para->u32ClassNum * sizeof(HI_FLOAT);
    HI_U32 size_proposal = para->u32MaxRois * SVP_WK_PROPOSAL_WIDTH * sizeof(HI_U32);
    HI_U32 size_stack    = MAX_STACK_DEPTH * sizeof(NNIE_STACK_S);

    HI_U32 total_size    = size_scores + size_proposal + size_stack;

    return total_size;
}

HI_S32 write_result(Faster_RCNN_Para* para, HI_S32* pstDstScore, HI_S32* pstDstBbox,
    HI_S32* pstRoiOutCnt, std::string& resultPath, SVP_SAMPLE_FILE_NAME_PAIR& imgNamePair)
{
    HI_U32 ClassNum       = para->u32ClassNum;
    HI_U32 u32MaxRoi      = para->u32MaxRois;
    HI_U32 u32ScoreBias   = 0;
    HI_U32 u32BboxBias    = 0;

    HI_U32 u32Index       = 0;
    HI_S32 s32XMin        = 0;
    HI_S32 s32YMin        = 0;
    HI_S32 s32XMax        = 0;
    HI_S32 s32YMax        = 0;
    HI_S32* ps32Score     = NULL;
    HI_S32* ps32Bbox      = NULL;
    HI_S32* ps32RoiOutCnt = NULL;
    HI_FLOAT f32Score     = 0.0;

    HI_BOOL hasResultOut  = HI_FALSE;

    stringstream ss;

    string pre_tmp = "roi_pos_out_";
    string post_tmp = ".txt";

    /* e.g. result_FASTER_RCNN_ALEX/000110_roi_pos_out_ALL.txt */
    string fileName = resultPath + imgNamePair.first + "_" + pre_tmp + "ALL" + post_tmp;
    ofstream foutALL(fileName.c_str());
    SVP_CHECK(foutALL.good(), HI_FAILURE);

    ps32Score = pstDstScore;
    ps32Bbox = pstDstBbox;
    ps32RoiOutCnt = pstRoiOutCnt;

    PrintBreakLine(HI_TRUE);

    ss << para->u32OriImWidth << "  " << para->u32OriImHeight << endl;
    cout << ss.str();
    foutALL << ss.str();
    ss.str("");

    for (HI_U32 i = 0; i < ClassNum; i++)
    {
        /* record info by class */
        fileName = resultPath + imgNamePair.first + "_" + pre_tmp + std::to_string((HI_U64)i) + post_tmp;
        ofstream fout(fileName.c_str());
        SVP_CHECK(fout.good(), HI_FAILURE);

        u32ScoreBias = i * u32MaxRoi;
        u32BboxBias  = i * u32MaxRoi * SVP_WK_COORDI_NUM;

        for (u32Index = 0; u32Index < (HI_U32)ps32RoiOutCnt[i]; u32Index++)
        {
            f32Score = (HI_FLOAT)ps32Score[u32ScoreBias + u32Index] / SVP_WK_QUANT_BASE;
            if (f32Score > 1.0 || f32Score < 0.0) {
                printf("ERROR in f32Score(%f), out of range[0,1]\n", f32Score);
            }
            else {
                s32XMin = SizeClip(ps32Bbox[u32BboxBias + u32Index*SVP_WK_COORDI_NUM + 0], 0, (HI_S32)(para->u32OriImWidth - 1));
                s32YMin = SizeClip(ps32Bbox[u32BboxBias + u32Index*SVP_WK_COORDI_NUM + 1], 0, (HI_S32)(para->u32OriImHeight - 1));
                s32XMax = SizeClip(ps32Bbox[u32BboxBias + u32Index*SVP_WK_COORDI_NUM + 2], 0, (HI_S32)(para->u32OriImWidth - 1));
                s32YMax = SizeClip(ps32Bbox[u32BboxBias + u32Index*SVP_WK_COORDI_NUM + 3], 0, (HI_S32)(para->u32OriImHeight - 1));

                ss << imgNamePair.first<< "  " <<
                    setw(4) << i <<"  " <<
                    fixed << setprecision(8) << f32Score << "  " <<
                    setw(4) << s32XMin << "  " <<
                    setw(4) << s32YMin << "  " <<
                    setw(4) << s32XMax << "  " <<
                    setw(4) << s32YMax << endl;

                hasResultOut = HI_TRUE;

                /* dump to roi_pos_out_ALL.txt except class 0*/
                if (0 != i) 
                {
                    foutALL << ss.str();
                }

                /* dump to roi_pos_out_i.txt */
                fout << ss.str();
                cout << ss.str();
                ss.str("");
            }
        }
        fout.close();

        PrintBreakLine(hasResultOut);
        hasResultOut = HI_FALSE;
    }
    foutALL.close();

    return HI_SUCCESS;
}

HI_S32 Print_Result1(HI_CHAR* filename, HI_CHAR* buff, HI_U32 data_len, HI_U32 data_with)
{
    ofstream fout(filename);
    SVP_CHECK(fout.good(), HI_FAILURE);

    for (HI_U32 i = 0; i < data_len; i = i + data_with)
    {
        if (1 == data_with)
        {
            fout << (HI_CHAR)*((HI_CHAR*)(buff + i)) << endl;
        }
        else if (2 == data_with)
        {
            fout << (HI_S16)*((HI_S16*)(buff + i)) << endl;
        }
        else if (4 == data_with)
        {
            fout << (HI_S32)*((HI_S32*)(buff + i)) << endl;
        }
        else
        {
            printf("Error!\n");
            return HI_FAILURE;
        }

    }
    fout.close();

    return HI_SUCCESS;
}
