#include "detectionCom.h"
#include "ssd_software.h"
#include <stdio.h>


/*******************************************************
PriorBox Function
*******************************************************/
HI_S32 PriorBoxForward(
    HI_U32 u32LayerWidth,
    HI_U32 u32LayerHeight,
    HI_U32 u32ImgWidth,
    HI_U32 u32ImgHeight,
    HI_FLOAT f32MinSize,
    HI_U32 u32MinSizeNum,
    HI_FLOAT f32MaxSize,
    HI_U32 u32MaxSizeNum,
    HI_U32 u32Flip,
    HI_U32 u32Clip,
    HI_U32 u32InputARNum,
    HI_FLOAT* au32AspectRatio,
    HI_FLOAT f32StepW,
    HI_FLOAT f32StepH,
    HI_FLOAT f32Offset,
    HI_S32* au32Var,
    HI_S32* s32OutputData
)
{
    HI_U32 u32AspectRatioNum    = 0;
    HI_U32 u32Index             = 0;
    HI_FLOAT af32AspectRatio[6] = { 0 };
    HI_U32 u32NumPrior          = 0;

    HI_FLOAT f32CenterX     = 0;
    HI_FLOAT f32CenterY     = 0;
    HI_FLOAT f32BoxHeight   = 0;
    HI_FLOAT f32BoxWidth    = 0;
    HI_FLOAT f32MaxBoxWidth = 0;

    HI_FLOAT af32BoxWidthAR[6]  = { 0 };
    HI_FLOAT af32BoxHeightAR[6] = { 0 };

    HI_U32 i = 0;
    HI_U32 j = 0;
    HI_U32 h = 0;
    HI_U32 w = 0;

    // generate aspect_ratios
    u32AspectRatioNum = 0;
    af32AspectRatio[0] = 1;
    u32AspectRatioNum++;

    for (i = 0; i < (HI_U32)u32InputARNum; i++)
    {
        af32AspectRatio[u32AspectRatioNum++] = (HI_FLOAT)au32AspectRatio[i];
        if (u32Flip)
        {
            af32AspectRatio[u32AspectRatioNum++] = (HI_FLOAT)1 / (HI_FLOAT)au32AspectRatio[i];
        }
    }

    u32NumPrior = u32MinSizeNum * u32AspectRatioNum + u32MaxSizeNum;
    f32MaxBoxWidth = (HI_FLOAT)sqrt(f32MinSize * f32MaxSize);

    for (i = 0; i < u32AspectRatioNum; i++)
    {
        af32BoxWidthAR[i] = f32MinSize * (HI_FLOAT)sqrt(af32AspectRatio[i]);
        af32BoxHeightAR[i] = f32MinSize / (HI_FLOAT)sqrt(af32AspectRatio[i]);
    }

    u32Index = 0;
    for (h = 0; h < u32LayerHeight; h++)
    {
        for (w = 0; w < u32LayerWidth; w++)
        {
            f32CenterX = (w + f32Offset) * f32StepW;
            f32CenterY = (h + f32Offset) * f32StepH;

            /*** first prior ***/
            f32BoxHeight = f32MinSize;
            f32BoxWidth = f32MinSize;

            s32OutputData[u32Index++] = (HI_S32)(f32CenterX - f32BoxWidth * 0.5f);
            s32OutputData[u32Index++] = (HI_S32)(f32CenterY - f32BoxHeight * 0.5f);
            s32OutputData[u32Index++] = (HI_S32)(f32CenterX + f32BoxWidth * 0.5f);
            s32OutputData[u32Index++] = (HI_S32)(f32CenterY + f32BoxHeight * 0.5f);

            /*** second prior ***/
            f32BoxHeight = f32MaxBoxWidth;
            f32BoxWidth = f32MaxBoxWidth;

            s32OutputData[u32Index++] = (HI_S32)(f32CenterX - f32BoxWidth * 0.5f);
            s32OutputData[u32Index++] = (HI_S32)(f32CenterY - f32BoxHeight * 0.5f);

            s32OutputData[u32Index++] = (HI_S32)(f32CenterX + f32BoxWidth * 0.5f);
            s32OutputData[u32Index++] = (HI_S32)(f32CenterY + f32BoxHeight * 0.5f);

            /**** rest of priors, skip AspectRatio == 1 ****/
            for (i = 1; i < u32AspectRatioNum; i++)
            {
                f32BoxWidth = af32BoxWidthAR[i];
                f32BoxHeight = af32BoxHeightAR[i];
                s32OutputData[u32Index++] = (HI_S32)(f32CenterX - f32BoxWidth * 0.5f);
                s32OutputData[u32Index++] = (HI_S32)(f32CenterY - f32BoxHeight * 0.5f);
                s32OutputData[u32Index++] = (HI_S32)(f32CenterX + f32BoxWidth * 0.5f);
                s32OutputData[u32Index++] = (HI_S32)(f32CenterY + f32BoxHeight * 0.5f);
            }

        } // end of for( LayerWidth )
    } // end of for( LayerHeight )

    /************ clip the priors' coordinates, within [0, u32ImgWidth] & [0, u32ImgHeight] *************/
    if (u32Clip)
    {
        for (i = 0; i < (HI_U32)(u32LayerWidth * u32LayerHeight * u32NumPrior / 2); i++)
        {
            s32OutputData[2 * i] = SizeClip(s32OutputData[2 * i], 0, (HI_S32)u32ImgWidth);
            s32OutputData[2 * i + 1] = SizeClip(s32OutputData[2 * i + 1], 0, (HI_S32)u32ImgHeight);
        }
    }

    /*********************** get var **********************/
    // check var num
    for (h = 0; h < u32LayerHeight; h++)
    {
        for (w = 0; w < u32LayerWidth; w++)
        {
            for (i = 0; i < u32NumPrior; i++)
            {
                for (j = 0; j < 4; j++)
                {
                    s32OutputData[u32Index++] = (HI_S32)au32Var[j];
                }
            }
        }
    }

    return HI_SUCCESS;
}

/**********************************************************
Softmax sub function
**********************************************************/
HI_S32 SSD_SoftMax(
    HI_S32* s32Src,
    HI_S32* s32Dst,
    HI_S32 s32ArraySize)
{
    /***** define parameters ****/
    HI_S32 s32Max = 0;
    HI_S32 s32Sum = 0;
    HI_S32 i = 0;

    for (i = 0; i < s32ArraySize; ++i)
    {
        if (s32Max < s32Src[i])
        {
            s32Max = s32Src[i];
        }
    }

    for (i = 0; i < s32ArraySize; ++i)
    {
        s32Dst[i] = (HI_S32)(SVP_WK_QUANT_BASE * exp((HI_FLOAT)(s32Src[i] - s32Max) / SVP_WK_QUANT_BASE));
        s32Sum += s32Dst[i];
    }

    for (i = 0; i < s32ArraySize; ++i)
    {
        s32Dst[i] = (HI_S32)(((HI_FLOAT)s32Dst[i] / (HI_FLOAT)s32Sum) * SVP_WK_QUANT_BASE);
    }
    return HI_SUCCESS;
}

/***************************************************
Forward Function
****************************************************/
HI_S32 SoftmaxForward(
    // input parameters
    HI_U32 u32InputHeight,
    HI_U32* au32InputChannel,
    HI_U32 u32ConcatNum,

    // output parameters
    HI_S32* input_width,

    HI_U32 u32OutputWidth,
    HI_U32 u32OutputHeight,
    HI_U32 u32OutputChannel,

    HI_S32** as32InputData,
    HI_S32* s32OutputData
)
{
    HI_S32* s32InputData = NULL;
    HI_S32* s32OutputTmp = NULL;

    HI_U32 u32OuterNum = 0;
    HI_U32 u32InnerNum = 0;
    HI_U32 u32InputChannel = 0;

    HI_U32 i = 0;

    HI_U32 u32ConcatCnt = 0;
    HI_S32 s32Ret = 0;
    HI_S32 input_stride = 0;
    HI_U32 skip = 0;
    HI_U32 left = 0;

    s32OutputTmp = s32OutputData;
    for (u32ConcatCnt = 0; u32ConcatCnt < u32ConcatNum; u32ConcatCnt++)
    {
        s32InputData = (HI_S32*)as32InputData[u32ConcatCnt];
        input_stride = input_width[u32ConcatCnt];

        u32InputChannel = au32InputChannel[u32ConcatCnt];

        u32OuterNum = u32InputChannel / u32InputHeight;
        u32InnerNum = u32InputHeight;

        skip = (HI_U32)input_stride / u32InnerNum;
        left = (HI_U32)input_stride % u32InnerNum;

        // do softmax
        for (i = 0; i < u32OuterNum; i++)
        {
            s32Ret = SSD_SoftMax(s32InputData, s32OutputTmp, (HI_S32)u32InnerNum);
            if ((i + 1) % skip == 0)
            {
                s32InputData += left;
            }

            s32InputData += u32InnerNum;
            s32OutputTmp += u32InnerNum;
        }
    }

    return s32Ret;
}

HI_S32 DetectionOutForward(
    HI_U32 u32BottomSize, // concat num
    HI_S32 s32ConfThresh, // Confidence thresh
    HI_U32 u32NumClasses,
    HI_U32 u32TopK,
    HI_U32 u32KeepTopK,
    HI_U32 u32NmsThresh,
    HI_U32* au32InputChannel,

    // --- loc data ----
    HI_S32** as32AllLocPreds,
    HI_S32** as32AllPriorBoxes,
    HI_S32* s32ConfScores,
    HI_S32* pstAssistMemPool,

    // ---- final result -----
    HI_S32* ps32DstScoreSrc,
    HI_S32* ps32DstBboxSrc,
    HI_S32* ps32RoiOutCntSrc

)
{

    /************* check input parameters ****************/
    /******** define variables **********/
    HI_S32* s32LocPreds          = NULL;
    HI_S32* s32PriorBoxes        = NULL;
    HI_S32* s32PriorVar          = NULL;

    HI_S32* s32AllDecodeBoxes    = NULL;

    HI_S32* ps32DstScore         = NULL;
    HI_S32* ps32DstBbox          = NULL;
    HI_S32* ps32RoiOutCnt        = NULL;
    HI_U32 u32RoiOutCnt          = 0;

    HI_S32* s32SingleProposal    = NULL;
    HI_S32* ps32AfterTopK        = NULL;
    NNIE_STACK_S* pstStack       = NULL;

    HI_U32 u32PriorNum           = 0;
    HI_U32 u32NumPredsPerClass   = 0;

    HI_FLOAT f32PriorWidth       = 0;
    HI_FLOAT f32PriorHeight      = 0;
    HI_FLOAT f32PriorCenterX     = 0;
    HI_FLOAT f32PriorCenterY     = 0;

    HI_FLOAT f32DecodeBoxCenterX = 0;
    HI_FLOAT f32DecodeBoxCenterY = 0;
    HI_FLOAT f32DecodeBoxWidth   = 0;
    HI_FLOAT f32DecodeBoxHeight  = 0;

    HI_U32 u32SrcIdx             = 0;
    HI_U32 u32AfterFilter        = 0;
    HI_U32 u32AfterTopK          = 0;
    HI_U32 u32KeepCnt            = 0;
    HI_U32 i = 0;
    HI_U32 j = 0;


    HI_S32 s32Ret = HI_FAILURE;

    u32PriorNum = 0;
    for (i = 0; i < u32BottomSize; i++)
    {
        u32PriorNum += au32InputChannel[i] / 4;
    }

    // ----- prepare for Assist MemPool ----
    s32AllDecodeBoxes = (HI_S32*)pstAssistMemPool;
    s32SingleProposal = s32AllDecodeBoxes + u32PriorNum * 4;
    ps32AfterTopK = s32SingleProposal + 6 * u32PriorNum;
    pstStack = (NNIE_STACK_S*)(ps32AfterTopK + u32PriorNum * 6);

    u32SrcIdx = 0;

    for (i = 0; i < u32BottomSize; i++) // u32BottomSize, the number of CONCAT
    {
        /********** get loc predictions ************/
        s32LocPreds = as32AllLocPreds[i];

        u32NumPredsPerClass = au32InputChannel[i] / 4;

        /********** get Prior Bboxes ************/
        s32PriorBoxes = (HI_S32*)as32AllPriorBoxes[i];

        s32PriorVar = s32PriorBoxes + u32NumPredsPerClass * 4;

        for (j = 0; j < u32NumPredsPerClass; j++)
        {
            f32PriorWidth   = (HI_FLOAT)(s32PriorBoxes[j * 4 + 2] - s32PriorBoxes[j * 4]);
            f32PriorHeight  = (HI_FLOAT)(s32PriorBoxes[j * 4 + 3] - s32PriorBoxes[j * 4 + 1]);
            f32PriorCenterX = (s32PriorBoxes[j * 4 + 2] + s32PriorBoxes[j * 4]) * 0.5f;
            f32PriorCenterY = (s32PriorBoxes[j * 4 + 3] + s32PriorBoxes[j * 4 + 1]) * 0.5f;

            f32DecodeBoxCenterX = ((HI_FLOAT)s32PriorVar[j * 4] / SVP_WK_QUANT_BASE) * ((HI_FLOAT)s32LocPreds[j * 4] / SVP_WK_QUANT_BASE) * f32PriorWidth + f32PriorCenterX;
            f32DecodeBoxCenterY = ((HI_FLOAT)s32PriorVar[j * 4 + 1] / SVP_WK_QUANT_BASE) * ((HI_FLOAT)s32LocPreds[j * 4 + 1] / SVP_WK_QUANT_BASE) * f32PriorHeight + f32PriorCenterY;
            f32DecodeBoxWidth   = (HI_FLOAT)exp(((HI_FLOAT)s32PriorVar[j * 4 + 2] / SVP_WK_QUANT_BASE) * ((HI_FLOAT)s32LocPreds[j * 4 + 2] / SVP_WK_QUANT_BASE)) * f32PriorWidth;
            f32DecodeBoxHeight  = (HI_FLOAT)exp(((HI_FLOAT)s32PriorVar[j * 4 + 3] / SVP_WK_QUANT_BASE) * ((HI_FLOAT)s32LocPreds[j * 4 + 3] / SVP_WK_QUANT_BASE)) * f32PriorHeight;

            s32AllDecodeBoxes[u32SrcIdx++] = (HI_S32)(f32DecodeBoxCenterX - f32DecodeBoxWidth * 0.5f);
            s32AllDecodeBoxes[u32SrcIdx++] = (HI_S32)(f32DecodeBoxCenterY - f32DecodeBoxHeight * 0.5f);
            s32AllDecodeBoxes[u32SrcIdx++] = (HI_S32)(f32DecodeBoxCenterX + f32DecodeBoxWidth * 0.5f);
            s32AllDecodeBoxes[u32SrcIdx++] = (HI_S32)(f32DecodeBoxCenterY + f32DecodeBoxHeight * 0.5f);
        }

    } // end of CONCAT

    /********** do NMS for each class *************/
    u32AfterTopK = 0;
    for (i = 0; i < u32NumClasses; i++) // classification num, 21 for PASCAL VOC
    {
        for (j = 0; j < u32PriorNum; j++)
        {
            s32SingleProposal[j * SVP_WK_PROPOSAL_WIDTH + 0] = s32AllDecodeBoxes[j * SVP_WK_COORDI_NUM + 0];
            s32SingleProposal[j * SVP_WK_PROPOSAL_WIDTH + 1] = s32AllDecodeBoxes[j * SVP_WK_COORDI_NUM + 1];
            s32SingleProposal[j * SVP_WK_PROPOSAL_WIDTH + 2] = s32AllDecodeBoxes[j * SVP_WK_COORDI_NUM + 2];
            s32SingleProposal[j * SVP_WK_PROPOSAL_WIDTH + 3] = s32AllDecodeBoxes[j * SVP_WK_COORDI_NUM + 3];
            s32SingleProposal[j * SVP_WK_PROPOSAL_WIDTH + 4] = s32ConfScores[j*u32NumClasses + i];
            s32SingleProposal[j * SVP_WK_PROPOSAL_WIDTH + 5] = 0;
        }

        s32Ret = NonRecursiveArgQuickSort(s32SingleProposal, 0, (HI_S32)u32PriorNum - 1, pstStack);
        SVP_CHECK(HI_SUCCESS == s32Ret, HI_FAILURE);

        u32AfterFilter = (u32PriorNum < u32TopK) ? u32PriorNum : u32TopK;

        s32Ret = NonMaxSuppression(s32SingleProposal, u32AfterFilter, u32NmsThresh);
        SVP_CHECK(HI_SUCCESS == s32Ret, HI_FAILURE);

        u32RoiOutCnt  = 0;
        ps32DstScore  = (HI_S32*)ps32DstScoreSrc;
        ps32DstBbox   = (HI_S32*)ps32DstBboxSrc;
        ps32RoiOutCnt = (HI_S32*)ps32RoiOutCntSrc;

        ps32DstScore += (HI_S32)(i * u32TopK);
        ps32DstBbox  += (HI_S32)(i * u32TopK * 4);

        for (j = 0; j < u32TopK; j++)
        {
            if (s32SingleProposal[j * SVP_WK_PROPOSAL_WIDTH + 5] == 0 &&
                s32SingleProposal[j * SVP_WK_PROPOSAL_WIDTH + 4] > s32ConfThresh)
            {
                ps32DstScore[u32RoiOutCnt] = s32SingleProposal[j * SVP_WK_PROPOSAL_WIDTH + 4];
                ps32DstBbox[u32RoiOutCnt * SVP_WK_COORDI_NUM + 0] = s32SingleProposal[j * SVP_WK_PROPOSAL_WIDTH + 0];
                ps32DstBbox[u32RoiOutCnt * SVP_WK_COORDI_NUM + 1] = s32SingleProposal[j * SVP_WK_PROPOSAL_WIDTH + 1];
                ps32DstBbox[u32RoiOutCnt * SVP_WK_COORDI_NUM + 2] = s32SingleProposal[j * SVP_WK_PROPOSAL_WIDTH + 2];
                ps32DstBbox[u32RoiOutCnt * SVP_WK_COORDI_NUM + 3] = s32SingleProposal[j * SVP_WK_PROPOSAL_WIDTH + 3];
                u32RoiOutCnt++;
            }
        }
        ps32RoiOutCnt[i] = (HI_S32)u32RoiOutCnt;

        u32AfterTopK += u32RoiOutCnt;
    }

    u32KeepCnt = 0;
    if (u32AfterTopK > u32KeepTopK)
    {
        for (i = 1; i < u32NumClasses; i++)
        {
            ps32DstScore  = ps32DstScoreSrc;
            ps32DstBbox   = ps32DstBboxSrc;
            ps32RoiOutCnt = ps32RoiOutCntSrc;

            ps32DstScore += (HI_S32)(i * u32TopK);
            ps32DstBbox  += (HI_S32)(i * u32TopK * SVP_WK_COORDI_NUM);

            for (j = 0; j < (HI_U32)ps32RoiOutCnt[i]; j++)
            {
                ps32AfterTopK[u32KeepCnt * SVP_WK_PROPOSAL_WIDTH + 0] = ps32DstBbox[j * SVP_WK_COORDI_NUM + 0];
                ps32AfterTopK[u32KeepCnt * SVP_WK_PROPOSAL_WIDTH + 1] = ps32DstBbox[j * SVP_WK_COORDI_NUM + 1];
                ps32AfterTopK[u32KeepCnt * SVP_WK_PROPOSAL_WIDTH + 2] = ps32DstBbox[j * SVP_WK_COORDI_NUM + 2];
                ps32AfterTopK[u32KeepCnt * SVP_WK_PROPOSAL_WIDTH + 3] = ps32DstBbox[j * SVP_WK_COORDI_NUM + 3];
                ps32AfterTopK[u32KeepCnt * SVP_WK_PROPOSAL_WIDTH + 4] = ps32DstScore[j];
                ps32AfterTopK[u32KeepCnt * SVP_WK_PROPOSAL_WIDTH + 5] = i;
                u32KeepCnt++;
            }
        }

        s32Ret = NonRecursiveArgQuickSort(ps32AfterTopK, 0, (HI_S32)u32KeepCnt - 1, pstStack);
        SVP_CHECK(HI_SUCCESS == s32Ret, HI_FAILURE);

        for (i = 1; i < u32NumClasses; i++)
        {
            u32RoiOutCnt  = 0;
            ps32DstScore  = (HI_S32*)ps32DstScoreSrc;
            ps32DstBbox   = (HI_S32*)ps32DstBboxSrc;
            ps32RoiOutCnt = (HI_S32*)ps32RoiOutCntSrc;

            ps32DstScore += (HI_S32)(i * u32TopK);
            ps32DstBbox  += (HI_S32)(i * u32TopK * SVP_WK_COORDI_NUM);

            for (j = 0; j < u32KeepTopK; j++)
            {
                if (ps32AfterTopK[j * SVP_WK_PROPOSAL_WIDTH + 5] == (HI_S32)i)
                {
                    ps32DstScore[u32RoiOutCnt] = ps32AfterTopK[j * SVP_WK_PROPOSAL_WIDTH + 4];
                    ps32DstBbox[u32RoiOutCnt * SVP_WK_COORDI_NUM + 0] = ps32AfterTopK[j * SVP_WK_PROPOSAL_WIDTH + 0];
                    ps32DstBbox[u32RoiOutCnt * SVP_WK_COORDI_NUM + 1] = ps32AfterTopK[j * SVP_WK_PROPOSAL_WIDTH + 1];
                    ps32DstBbox[u32RoiOutCnt * SVP_WK_COORDI_NUM + 2] = ps32AfterTopK[j * SVP_WK_PROPOSAL_WIDTH + 2];
                    ps32DstBbox[u32RoiOutCnt * SVP_WK_COORDI_NUM + 3] = ps32AfterTopK[j * SVP_WK_PROPOSAL_WIDTH + 3];
                    u32RoiOutCnt++;
                }
            }
            ps32RoiOutCnt[i] = (HI_S32)u32RoiOutCnt;
        }
    }

    return HI_SUCCESS;
}
