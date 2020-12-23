#include <math.h>
#include <sstream>

#include "SvpSampleWk.h"
#include "SvpSampleCom.h"
#include "SvpSampleYolo.h"

#include "yolo_interface.h"

//
#define SVP_SAMPLE_YOLOV3_SCORE_FILTER_THREASH     (0.5f)
#define SVP_SAMPLE_YOLOV3_NMS_THREASH              (0.45f)

static HI_DOUBLE s_SvpSampleYoloV3Bias[SVP_SAMPLE_YOLOV3_SCALE_TYPE_MAX][6] = {
    {116,90, 156,198, 373,326},
    {30,61, 62,45, 59,119},
    {10,13, 16,30, 33,23}
};

HI_U32 SvpSampleWkYoloV3GetGridNum(SVP_SAMPLE_YOLOV3_SCALE_TYPE_E enScaleType)
{
    switch (enScaleType)
    {
    case CONV_82:
        return SVP_SAMPLE_YOLOV3_GRIDNUM_CONV_82;
        break;
    case CONV_94:
        return SVP_SAMPLE_YOLOV3_GRIDNUM_CONV_94;
        break;
    case CONV_106:
        return SVP_SAMPLE_YOLOV3_GRIDNUM_CONV_106;
        break;
    default:
        return 0;
    }
}

HI_U32 SvpSampleWkYoloV3GetBoxTotleNum(SVP_SAMPLE_YOLOV3_SCALE_TYPE_E enScaleType)
{
    switch (enScaleType)
    {
    case CONV_82:
        return SVP_SAMPLE_YOLOV3_GRIDNUM_CONV_82 * SVP_SAMPLE_YOLOV3_GRIDNUM_CONV_82 * SVP_SAMPLE_YOLOV3_BOXNUM;
    case CONV_94:
        return SVP_SAMPLE_YOLOV3_GRIDNUM_CONV_94 * SVP_SAMPLE_YOLOV3_GRIDNUM_CONV_94 * SVP_SAMPLE_YOLOV3_BOXNUM;
    case CONV_106:
        return SVP_SAMPLE_YOLOV3_GRIDNUM_CONV_106 * SVP_SAMPLE_YOLOV3_GRIDNUM_CONV_82 * SVP_SAMPLE_YOLOV3_BOXNUM;
    default:
        return 0;
    }
}

HI_U32 SvpSampleGetYolov3ResultMemSize(SVP_SAMPLE_YOLOV3_SCALE_TYPE_E enScaleType)
{
    HI_U32 u32GridNum = SvpSampleWkYoloV3GetGridNum(enScaleType);
    HI_U32 inputdate_size = u32GridNum * u32GridNum * SVP_SAMPLE_YOLOV3_CHANNLENUM * sizeof(HI_FLOAT);
    HI_U32 u32TmpBoxSize = u32GridNum * u32GridNum * SVP_SAMPLE_YOLOV3_CHANNLENUM * sizeof(HI_U32);
    HI_U32 u32BoxSize = u32GridNum * u32GridNum * SVP_SAMPLE_YOLOV3_BOXNUM * SVP_SAMPLE_YOLOV3_CLASSNUM * sizeof(SVP_SAMPLE_BOX_S);
    return (inputdate_size + u32TmpBoxSize + u32BoxSize);

}

void SvpSampleWkYoloV3GetResultForOneBlob(SVP_BLOB_S *pstDstBlob,
                                            HI_U8* pu8InputData,
                                            SVP_SAMPLE_YOLOV3_SCALE_TYPE_E enScaleType,
                                            HI_S32 *ps32ResultMem,
                                            SVP_SAMPLE_BOX_S** ppstBox,
                                            HI_U32 *pu32BoxNum)
{
    // result calc para config
    HI_FLOAT f32ScoreFilterThresh = SVP_SAMPLE_YOLOV3_SCORE_FILTER_THREASH;

    HI_U32 u32GridNum = SvpSampleWkYoloV3GetGridNum(enScaleType);
    HI_U32 u32CStep = u32GridNum * u32GridNum;
    HI_U32 u32HStep = u32GridNum;

    HI_U32 inputdate_size = u32GridNum * u32GridNum * SVP_SAMPLE_YOLOV3_CHANNLENUM;
    HI_U32 u32TmpBoxSize = u32GridNum * u32GridNum * SVP_SAMPLE_YOLOV3_CHANNLENUM;

    HI_FLOAT* pf32InputData = (HI_FLOAT*)ps32ResultMem;
    HI_FLOAT* pf32BoxTmp = (HI_FLOAT*)(pf32InputData + inputdate_size);////tep_box_size
    SVP_SAMPLE_BOX_S* pstBox = (SVP_SAMPLE_BOX_S*)(pf32BoxTmp + u32TmpBoxSize);////assit_box_size

    printf("n:%u, c:%u, h:%u, w:%u\n",
        pstDstBlob->u32Num,
        pstDstBlob->unShape.stWhc.u32Chn,
        pstDstBlob->unShape.stWhc.u32Height,
        pstDstBlob->unShape.stWhc.u32Width);

    if ((u32GridNum != pstDstBlob->unShape.stWhc.u32Height) ||
        (u32GridNum != pstDstBlob->unShape.stWhc.u32Width)  ||
        (SVP_SAMPLE_YOLOV3_CHANNLENUM != pstDstBlob->unShape.stWhc.u32Chn))
    {
        printf("error grid number!\n");
        return;
    }

    HI_U32 u32OneCSize = pstDstBlob->u32Stride * pstDstBlob->unShape.stWhc.u32Height;
    HI_U32 u32BoxsNum = 0;

    {
        HI_U32 n = 0;
        for (HI_U32 c = 0; c < SVP_SAMPLE_YOLOV3_CHANNLENUM; c++) {
            for (HI_U32 h = 0; h < u32GridNum; h++) {
                for (HI_U32 w = 0; w < u32GridNum; w++) {
                    HI_S32* ps32Temp = (HI_S32*)(pu8InputData + c * u32OneCSize + h * pstDstBlob->u32Stride) + w;
                    pf32InputData[n++] = (HI_FLOAT)(*ps32Temp) / SVP_WK_QUANT_BASE;
                }
            }
        }
    }
    {
        HI_U32 n = 0;
        for (HI_U32 h = 0; h < u32GridNum; h++) {
            for (HI_U32 w = 0; w < u32GridNum; w++) {
                for (HI_U32 c = 0; c < SVP_SAMPLE_YOLOV3_CHANNLENUM; c++) {
                    pf32BoxTmp[n++] = pf32InputData[c * u32CStep + h * u32HStep + w];
                }
            }
        }
    }

    for (HI_U32 n = 0; n < u32GridNum * u32GridNum; n++)
    {
        //Grid
        HI_U32 w = n % u32GridNum;
        HI_U32 h = n / u32GridNum;
        for (HI_U32 k = 0; k < SVP_SAMPLE_YOLOV3_BOXNUM; k++)
        {
            HI_U32 u32Index = (n * SVP_SAMPLE_YOLOV3_BOXNUM + k) * SVP_SAMPLE_YOLOV3_PARAMNUM;
            HI_FLOAT x = ((HI_FLOAT)w + Sigmoid(pf32BoxTmp[u32Index + 0])) / u32GridNum;
            HI_FLOAT y = ((HI_FLOAT)h + Sigmoid(pf32BoxTmp[u32Index + 1])) / u32GridNum;
            HI_FLOAT f32Width = (HI_FLOAT)(exp(pf32BoxTmp[u32Index + 2]) * s_SvpSampleYoloV3Bias[enScaleType][2 * k]) / SVP_SAMPLE_YOLOV3_SRC_WIDTH;
            HI_FLOAT f32Height = (HI_FLOAT)(exp(pf32BoxTmp[u32Index + 3]) * s_SvpSampleYoloV3Bias[enScaleType][2 * k + 1]) / SVP_SAMPLE_YOLOV3_SRC_HEIGHT;

            HI_FLOAT f32ObjScore = Sigmoid(pf32BoxTmp[u32Index + 4]); //objscore;
            if(f32ObjScore <= f32ScoreFilterThresh){
                continue;
            }

            for (HI_U32 classIdx = 0; classIdx < SVP_SAMPLE_YOLOV3_CLASSNUM; classIdx++)
            {
                HI_U32 u32ClassIdxBase = u32Index + 4 + 1;
                HI_FLOAT f32ClassScore = Sigmoid(pf32BoxTmp[u32ClassIdxBase + classIdx]); //objscore;
                HI_FLOAT f32Prob = f32ObjScore * f32ClassScore;
                f32Prob = (f32Prob > f32ScoreFilterThresh) ? f32Prob : 0.0f;

                pstBox[u32BoxsNum].f32Xmin = x - f32Width * 0.5f;  // xmin
                pstBox[u32BoxsNum].f32Xmax = x + f32Width * 0.5f;  // xmax
                pstBox[u32BoxsNum].f32Ymin = y - f32Height * 0.5f; // ymin
                pstBox[u32BoxsNum].f32Ymax = y + f32Height * 0.5f; // ymax
                pstBox[u32BoxsNum].f32ClsScore = f32Prob;          // predict prob
                pstBox[u32BoxsNum].u32MaxScoreIndex = classIdx;    // class score index
                pstBox[u32BoxsNum].u32Mask = 0;                    // Suppression mask

                u32BoxsNum++;
            }
        }
    }
    *ppstBox = pstBox;
    *pu32BoxNum = u32BoxsNum;

}

void SvpSampleWKYoloV3BoxPostProcess(SVP_SAMPLE_BOX_S* pstInputBbox, HI_U32 u32InputBboxNum,
    SVP_SAMPLE_BOX_S* pstResultBbox, HI_U32 *pu32BoxNum)
{
    HI_FLOAT f32NmsThresh = SVP_SAMPLE_YOLOV3_NMS_THREASH;
    HI_U32 u32MaxBoxNum = SVP_SAMPLE_YOLOV3_MAX_BOX_NUM;
    HI_U32 u32SrcWidth  = SVP_SAMPLE_YOLOV3_SRC_WIDTH;
    HI_U32 u32SrcHeight = SVP_SAMPLE_YOLOV3_SRC_HEIGHT;

    HI_U32 u32AssistStackNum = SvpSampleWkYoloV3GetBoxTotleNum(CONV_82)
                               + SvpSampleWkYoloV3GetBoxTotleNum(CONV_94)
                               + SvpSampleWkYoloV3GetBoxTotleNum(CONV_106);

    HI_U32 u32AssistStackSize = u32AssistStackNum * sizeof(SVP_SAMPLE_STACK_S);

    SVP_SAMPLE_STACK_S* pstAssistStack = (SVP_SAMPLE_STACK_S*)malloc(u32AssistStackSize);////assit_size
    if (NULL == pstAssistStack) {
        printf("Malloc fail, size %d!\n", u32AssistStackSize);
        return;
    }
    memset(pstAssistStack, 0, u32AssistStackSize);

    //quick_sort
    NonRecursiveArgQuickSortWithBox(pstInputBbox, 0, u32InputBboxNum - 1, pstAssistStack);
    free(pstAssistStack);

    //Nms
    SvpDetYoloNonMaxSuppression(pstInputBbox, u32InputBboxNum, f32NmsThresh, u32MaxBoxNum);

    //Get the result
    HI_U32 u32BoxResultNum = 0;
    for (HI_U32 n = 0; (n < u32InputBboxNum) && (u32BoxResultNum < u32MaxBoxNum); n++) {
        if (0 == pstInputBbox[n].u32Mask) {
            pstResultBbox[u32BoxResultNum].f32Xmin = SVP_SAMPLE_MAX(pstInputBbox[n].f32Xmin * u32SrcWidth, 0);
            pstResultBbox[u32BoxResultNum].f32Xmax = SVP_SAMPLE_MIN(pstInputBbox[n].f32Xmax * u32SrcWidth, u32SrcWidth);
            pstResultBbox[u32BoxResultNum].f32Ymax = SVP_SAMPLE_MIN(pstInputBbox[n].f32Ymax * u32SrcHeight, u32SrcHeight);
            pstResultBbox[u32BoxResultNum].f32Ymin = SVP_SAMPLE_MAX(pstInputBbox[n].f32Ymin * u32SrcHeight, 0);
            pstResultBbox[u32BoxResultNum].f32ClsScore = pstInputBbox[n].f32ClsScore;
            pstResultBbox[u32BoxResultNum].u32MaxScoreIndex = pstInputBbox[n].u32MaxScoreIndex;

            u32BoxResultNum++;
        }
    }

    printf("u32BoxResultNum: %d\n", u32BoxResultNum);
    if (0 == u32BoxResultNum) {
        return;
    }

    *pu32BoxNum += u32BoxResultNum;

}

// SvpSampleWkYoloV3GetResult(pstDetParam->astDst, pstDetParam->ps32ResultMem,
// &stBoxesInfo, p32BoxNum, strResultFolderDir, imgNameRecoder);
void SvpSampleWkYoloV3GetResult(SVP_BLOB_S *pstDstBlob, HI_S32 *ps32ResultMem, SVP_SAMPLE_BOX_RESULT_INFO_S *pstResultBoxInfo,
    HI_U32 *pu32BoxNum, string& strResultFolderDir, vector<SVP_SAMPLE_FILE_NAME_PAIR>& imgNameRecoder)
{
    SVP_SAMPLE_BOX_S* pstResultBbox = pstResultBoxInfo->pstBbox;
    HI_U32 u32ResultBoxNum = 0;
    SVP_SAMPLE_BOX_RESULT_INFO_S stTempBoxResultInfo;

    SVP_SAMPLE_BOX_S* pstTempBbox = NULL;
    HI_U32 u32TempBoxNum = 0;

    for (HI_U32 u32NumIndex = 0; u32NumIndex < pstDstBlob->u32Num; u32NumIndex++)
    {
        SVP_SAMPLE_BOX_S* apstBox[SVP_SAMPLE_YOLOV3_RESULT_BLOB_NUM] = { NULL };
        HI_U32 au32BoxNum[SVP_SAMPLE_YOLOV3_RESULT_BLOB_NUM] = { 0 };

        SVP_SAMPLE_RESULT_MEM_HEAD_S *pstHead = (SVP_SAMPLE_RESULT_MEM_HEAD_S *)ps32ResultMem;

        for (HI_U32 u32resBlobIdx = 0; u32resBlobIdx < SVP_SAMPLE_YOLOV3_RESULT_BLOB_NUM; u32resBlobIdx++)
        {
            SVP_BLOB_S* pstTempBlob = &pstDstBlob[u32resBlobIdx];
            // pstTempBlob->u32Stride��nnie�����feature��ÿ���������ֽ�����
            // u32OneCSize��nnie�����feature��ÿ��channel�������ֽ���
            HI_U32 u32OneCSize = pstTempBlob->u32Stride * pstTempBlob->unShape.stWhc.u32Height;
            HI_U32 u32FrameStride = u32OneCSize * pstTempBlob->unShape.stWhc.u32Chn;
            // ���ʵ�c��ͨ����h�е�w�е�Ԫ��
            HI_U8* pu8InputData = (HI_U8*)pstTempBlob->u64VirAddr + u32NumIndex * u32FrameStride;

            if (HI_NULL != pstHead)
            {
                SvpSampleWkYoloV3GetResultForOneBlob(pstTempBlob,
                    pu8InputData,
                    (SVP_SAMPLE_YOLOV3_SCALE_TYPE_E)pstHead->u32Type,
                    (HI_S32*)(pstHead + 1),
                    &apstBox[u32resBlobIdx], &au32BoxNum[u32resBlobIdx]);
            }

            if (u32resBlobIdx < SVP_SAMPLE_YOLOV3_RESULT_BLOB_NUM - 1) {
                pstHead = (SVP_SAMPLE_RESULT_MEM_HEAD_S*)((HI_U8 *)(pstHead + 1) + pstHead->u32Len);
            }
        }

        u32TempBoxNum = au32BoxNum[0] + au32BoxNum[1] + au32BoxNum[2];
        pstTempBbox = (SVP_SAMPLE_BOX_S*)malloc(sizeof(SVP_SAMPLE_BOX_S) * u32TempBoxNum);
        memcpy((HI_U8*)pstTempBbox, (HI_U8*)apstBox[0], sizeof(SVP_SAMPLE_BOX_S) * au32BoxNum[0]);
        memcpy((HI_U8*)(pstTempBbox + au32BoxNum[0]), (HI_U8*)apstBox[1], sizeof(SVP_SAMPLE_BOX_S) * au32BoxNum[1]);
        memcpy((HI_U8*)(pstTempBbox + au32BoxNum[0] + au32BoxNum[1]), (HI_U8*)apstBox[2], sizeof(SVP_SAMPLE_BOX_S) * au32BoxNum[2]);

        SvpSampleWKYoloV3BoxPostProcess(pstTempBbox, u32TempBoxNum, &pstResultBbox[u32ResultBoxNum], &pu32BoxNum[u32NumIndex]);

        stTempBoxResultInfo.u32OriImHeight = pstResultBoxInfo->u32OriImHeight;
        stTempBoxResultInfo.u32OriImWidth = pstResultBoxInfo->u32OriImWidth;
        stTempBoxResultInfo.pstBbox = &pstResultBbox[u32ResultBoxNum];
        SvpDetYoloResultPrint(&stTempBoxResultInfo, pu32BoxNum[u32NumIndex], strResultFolderDir, imgNameRecoder[u32NumIndex]);

        u32ResultBoxNum = u32ResultBoxNum + pu32BoxNum[u32NumIndex];
        if (u32ResultBoxNum >= 1024)
        {
            printf("Box number reach max 1024!\n");
            free(pstTempBbox);
            return;
        }
        free(pstTempBbox);
    }
}
