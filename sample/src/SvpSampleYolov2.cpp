#include "SvpSampleWk.h"
#include "SvpSampleCom.h"
#include <math.h>

HI_DOUBLE g_SvpSampleYoloV2Bias[10] = {1.08,1.19,3.42,4.41,6.63,11.38,9.42,5.11,16.62,10.52};

void SVP_SAMPLE_BoxArgswap(SVP_SAMPLE_BOX_S* pstBox1, SVP_SAMPLE_BOX_S* pstBox2)
{
    SVP_SAMPLE_BOX_S stBoxTmp;

    memcpy(&stBoxTmp,pstBox1,sizeof(stBoxTmp));
    memcpy(pstBox1,pstBox2,sizeof(stBoxTmp));
    memcpy(pstBox2,&stBoxTmp,sizeof(stBoxTmp));
}

HI_S32 SAMPLE_SVP_NonRecursiveArgQuickSort(SVP_SAMPLE_BOX_S* pstBoxs, HI_S32 s32Low, HI_S32 s32High, SVP_SAMPLE_STACK_S *pstStack)
{
    HI_S32 i = s32Low;
    HI_S32 j = s32High;
    HI_S32 s32Top = 0;

    HI_FLOAT f32KeyConfidence = pstBoxs[s32Low].f32ClsScore;
    pstStack[s32Top].s32Min = s32Low;
    pstStack[s32Top].s32Max = s32High;

    while(s32Top > -1)
    {
        s32Low = pstStack[s32Top].s32Min;
        s32High = pstStack[s32Top].s32Max;
        i = s32Low;
        j = s32High;
        s32Top--;

        f32KeyConfidence = pstBoxs[s32Low].f32ClsScore;

        while(i < j)
        {
            while((i < j) && (f32KeyConfidence > pstBoxs[j].f32ClsScore))
            {
                j--;
            }
            if(i < j)
            {
                SVP_SAMPLE_BoxArgswap(&pstBoxs[i],&pstBoxs[j]);
                i++;
            }

            while((i < j) && (f32KeyConfidence < pstBoxs[i].f32ClsScore))
            {
                i++;
            }
            if(i < j)
            {
                SVP_SAMPLE_BoxArgswap(&pstBoxs[i],&pstBoxs[j]);
                j--;
            }
        }

        if(s32Low < i-1)
        {
            s32Top++;
            pstStack[s32Top].s32Min = s32Low;
            pstStack[s32Top].s32Max = i-1;
        }

        if(s32High > i+1)
        {
            s32Top++;
            pstStack[s32Top].s32Min = i+1;
            pstStack[s32Top].s32Max = s32High;
        }
    }
    return HI_SUCCESS;
}



HI_DOUBLE SVP_SAMPLE_Cal_Iou(SVP_SAMPLE_BOX_S *pstBox1,SVP_SAMPLE_BOX_S *pstBox2)
{
    HI_FLOAT InterWidth = 0.0;
    HI_FLOAT InterHeight = 0.0;
    HI_DOUBLE f64InterArea = 0.0;
    HI_DOUBLE f64Box1Area = 0.0;
    HI_DOUBLE f64Box2Area = 0.0;
    HI_DOUBLE f64UnionArea = 0.0;

    InterWidth =  SVP_SAMPLE_MIN(pstBox1->f32Xmax, pstBox2->f32Xmax) - SVP_SAMPLE_MAX(pstBox1->f32Xmin,pstBox2->f32Xmin);
    InterHeight = SVP_SAMPLE_MIN(pstBox1->f32Ymax, pstBox2->f32Ymax) - SVP_SAMPLE_MAX(pstBox1->f32Ymin,pstBox2->f32Ymin);

    if(InterWidth <= 0 || InterHeight <= 0)
    {
        return HI_SUCCESS;
    }

    f64InterArea = InterWidth * InterHeight;
    f64Box1Area = (pstBox1->f32Xmax - pstBox1->f32Xmin)* (pstBox1->f32Ymax - pstBox1->f32Ymin);
    f64Box2Area = (pstBox2->f32Xmax - pstBox2->f32Xmin)* (pstBox2->f32Ymax - pstBox2->f32Ymin);
    f64UnionArea = f64Box1Area + f64Box2Area - f64InterArea;

    return f64InterArea/f64UnionArea;
}

HI_S32 SAMPLE_SVP_NonMaxSuppression( SVP_SAMPLE_BOX_S* pstBoxs, HI_U32 u32BoxNum, HI_FLOAT f32NmsThresh,HI_U32 u32MaxRoiNum)
{
    HI_U32 i,j;
    HI_U32 u32Num = 0;
    HI_DOUBLE f64Iou = 0.0;

    for (i = 0; i < u32BoxNum && u32Num < u32MaxRoiNum; i++)
    {
        if(pstBoxs[i].u32Mask == 0 )
        {
            u32Num++;
            for(j= i+1;j< u32BoxNum; j++)
            {
                if( pstBoxs[j].u32Mask == 0 )
                {
                    f64Iou = SVP_SAMPLE_Cal_Iou(&pstBoxs[i],&pstBoxs[j]);
                    if(f64Iou >= (HI_DOUBLE)f32NmsThresh)
                    {
                        pstBoxs[j].u32Mask = 1;
                    }
                }
            }
        }
    }

    return HI_SUCCESS;
}

HI_S32 SAMPLE_SVP_SoftMax( HI_FLOAT* pf32Src, HI_U32 u32Num)
{
    HI_FLOAT f32Max = 0;
    HI_FLOAT f32Sum = 0;
    HI_U32 i = 0;

    for(i = 0; i < u32Num; ++i)
    {
        if(f32Max < pf32Src[i])
        {
            f32Max = pf32Src[i];
        }
    }
    for(i = 0; i < u32Num; ++i)
    {
        pf32Src[i] = (HI_FLOAT)(exp(pf32Src[i] - f32Max));
        f32Sum += pf32Src[i];
    }

    for(i = 0; i < u32Num; ++i)
    {
        pf32Src[i] /= f32Sum;
    }

    return HI_SUCCESS;
}


HI_FLOAT SAMPLE_SVP_Sigmoid( HI_FLOAT f32Val)
{
    return (HI_FLOAT)(1.0/(1 + exp(-f32Val)));
}

HI_FLOAT SvpSampleGetMaxVal(HI_FLOAT *pf32Val,HI_U32 u32Num,HI_U32 * pu32MaxValueIndex)
{
    HI_U32 i = 0;
    HI_FLOAT f32MaxTmp = 0;

    f32MaxTmp = pf32Val[0];
    *pu32MaxValueIndex = 0;
    for(i = 1;i < u32Num;i++)
    {
        if(pf32Val[i] > f32MaxTmp)
        {
            f32MaxTmp = pf32Val[i];
            *pu32MaxValueIndex = i;
        }
    }

    return f32MaxTmp;
}

void SvpSampleWkYoloV2GetResult(SVP_BLOB_S *pstDstBlob, HI_S32 *ps32ResultMem, SVP_SAMPLE_BOX_RESULT_INFO_S *pstBoxInfo,
    HI_U32 *pu32BoxNum, string& strResultFolderDir, vector<SVP_SAMPLE_FILE_NAME_PAIR>& imgNameRecoder)
{
    HI_U32 u32CStep = SVP_SAMPLE_YOLOV2_GRIDNUM * SVP_SAMPLE_YOLOV2_GRIDNUM;
    HI_U32 u32HStep = SVP_SAMPLE_YOLOV2_GRIDNUM;
    HI_U32 u32BoxsNum = 0;
    HI_FLOAT *pf32BoxTmp=NULL;
    HI_FLOAT *pf32InputData = NULL;
    //HI_FLOAT f32InputData[SVP_SAMPLE_YOLOV2_GRIDNUM*SVP_SAMPLE_YOLOV2_GRIDNUM*SVP_SAMPLE_YOLOV2_CHANNLENUM];
    HI_U32 inputdate_size = SVP_SAMPLE_YOLOV2_GRIDNUM*SVP_SAMPLE_YOLOV2_GRIDNUM*SVP_SAMPLE_YOLOV2_CHANNLENUM;
    pf32InputData = (HI_FLOAT*)ps32ResultMem;
    HI_FLOAT f32ObjScore = 0.0;
    HI_FLOAT f32MaxScore = 0.0;
    HI_U32 u32MaxValueIndex = 0;
    HI_U32 h, w, n = 0, c, k, u32Index;
    HI_FLOAT x, y, f32Width, f32Height;
    HI_U32 u32AssistStackNum = SVP_SAMPLE_YOLOV2_BOXTOTLENUM;
    HI_U32 u32AssitBoxNum = SVP_SAMPLE_YOLOV2_BOXTOTLENUM;
    HI_U32 u32TmpBoxSize = SVP_SAMPLE_YOLOV2_GRIDNUM*SVP_SAMPLE_YOLOV2_GRIDNUM*SVP_SAMPLE_YOLOV2_CHANNLENUM;
    HI_U32 u32BoxResultNum = 0;
    //SVP_MEM_INFO_S stAssistBuf, stBoxsBuff;
    //SVP_MEM_INFO_S stBoxTmpBuf;
    SVP_SAMPLE_STACK_S *pstAssistStack;
    SVP_SAMPLE_BOX_S *pstBox;
    HI_S32 *ps32InputData = NULL;

    HI_FLOAT f32NmsThresh = 0.3f;
    HI_U32 u32MaxBoxNum = SVP_SAMPLE_YOLOV2_MAX_BOX_NUM;
    HI_U32 u32SrcWidth = 416;
    HI_U32 u32SrcHeight = 416;
    HI_U32 u32OutBoxNum = 5;

    SVP_SAMPLE_BOX_S *pstBoxResult=NULL;

    //SvpSampleMallocMem(NULL, NULL, u32ResultBoxSize, &stBoxsResultBuff);
    //pstBoxResult = (SVP_SAMPLE_BOX_S*)stBoxsResultBuff.u64VirAddr;
    pf32BoxTmp = (HI_FLOAT*)(pf32InputData + inputdate_size);////tep_box_size
    pstBox = (SVP_SAMPLE_BOX_S*)(pf32BoxTmp + u32TmpBoxSize);////assit_box_size
    pstAssistStack = (SVP_SAMPLE_STACK_S*)(pstBox+ u32AssitBoxNum);////assit_size
    pstBoxResult = (SVP_SAMPLE_BOX_S*)(pstAssistStack + u32AssistStackNum);////result_box_size

    //malloc the assist buffer for sort
    //SvpSampleMallocMem(NULL, NULL, u32AssistBuffSize, &stAssistBuf);
    //pstAssistStack = (SVP_SAMPLE_STACK_S*)stAssistBuf.u64VirAddr;

    //malloc the assit box buffer
    //SvpSampleMallocMem(NULL, NULL, u32AssitBoxSize, &stBoxsBuff);
    //pstBox = (SVP_SAMPLE_BOX_S*)stBoxsBuff.u64VirAddr;

    //malloc the box tmp buffer
   // SvpSampleMallocMem(NULL, NULL, u32TmpBoxSize, &stBoxTmpBuf);


    HI_U32 u32FrameStride = pstDstBlob->u32Stride * pstDstBlob->unShape.stWhc.u32Chn * pstDstBlob->unShape.stWhc.u32Height;

    for (HI_U32 u32NumIndex = 0; u32NumIndex < pstDstBlob->u32Num; u32NumIndex++)
    {
        u32BoxResultNum = 0;
        u32BoxsNum = 0;
        ps32InputData = (HI_S32 *)((HI_U8*)pstDstBlob->u64VirAddr + u32NumIndex * u32FrameStride);
        //pf32BoxTmp = (HI_FLOAT*)stBoxTmpBuf.u64VirAddr;

        for (n = 0; n < SVP_SAMPLE_YOLOV2_BOXTOTLENUM*SVP_SAMPLE_YOLOV2_PARAMNUM; n++)
        {
            pf32InputData[n] = (HI_FLOAT)(ps32InputData[n] / 4096.0);
        }
        n = 0;

        for (h = 0; h < SVP_SAMPLE_YOLOV2_GRIDNUM; h++)
        {
            for (w = 0; w < SVP_SAMPLE_YOLOV2_GRIDNUM; w++)
            {
                for (c = 0; c < SVP_SAMPLE_YOLOV2_CHANNLENUM; c++)
                {
                    pf32BoxTmp[n++] = pf32InputData[c*u32CStep + h*u32HStep + w];
                }
            }
        }

        for (n = 0; n < SVP_SAMPLE_YOLOV2_GRIDNUM * SVP_SAMPLE_YOLOV2_GRIDNUM; n++)
        {
            //Grid
            w = n % SVP_SAMPLE_YOLOV2_GRIDNUM;
            h = n / SVP_SAMPLE_YOLOV2_GRIDNUM;
            for (k = 0; k < SVP_SAMPLE_YOLOV2_BOXNUM; k++)
            {
                u32Index = (n * SVP_SAMPLE_YOLOV2_BOXNUM + k) * SVP_SAMPLE_YOLOV2_PARAMNUM;
                x = (HI_FLOAT)(w + SAMPLE_SVP_Sigmoid(pf32BoxTmp[u32Index + 0])) / SVP_SAMPLE_YOLOV2_GRIDNUM;
                y = (HI_FLOAT)(h + SAMPLE_SVP_Sigmoid(pf32BoxTmp[u32Index + 1])) / SVP_SAMPLE_YOLOV2_GRIDNUM;
                f32Width  = (HI_FLOAT)(exp(pf32BoxTmp[u32Index + 2]) * g_SvpSampleYoloV2Bias[2 * k]) / SVP_SAMPLE_YOLOV2_GRIDNUM;
                f32Height = (HI_FLOAT)(exp(pf32BoxTmp[u32Index + 3]) * g_SvpSampleYoloV2Bias[2 * k + 1]) / SVP_SAMPLE_YOLOV2_GRIDNUM;

            f32ObjScore = SAMPLE_SVP_Sigmoid(pf32BoxTmp[u32Index + 4]); //objscore;
            SAMPLE_SVP_SoftMax(&pf32BoxTmp[u32Index + 5],SVP_SAMPLE_YOLOV2_CLASSNUM); // MaxClassScore;

            f32MaxScore = SvpSampleGetMaxVal(&pf32BoxTmp[u32Index + 5],SVP_SAMPLE_YOLOV2_CLASSNUM,&u32MaxValueIndex);

            if(f32MaxScore * f32ObjScore > 0.01) //&& width != 0 && height != 0) // filter the low score box
            {
                pstBox[u32BoxsNum].f32Xmin = (HI_FLOAT)(x - f32Width* 0.5);                // xmin
                pstBox[u32BoxsNum].f32Xmax = (HI_FLOAT)(x + f32Width* 0.5);                // xmax
                pstBox[u32BoxsNum].f32Ymin = (HI_FLOAT)(y - f32Height* 0.5);               // ymin
                pstBox[u32BoxsNum].f32Ymax = (HI_FLOAT)(y + f32Height* 0.5);               // ymax
                pstBox[u32BoxsNum].f32ClsScore = f32MaxScore * f32ObjScore; //class score
                pstBox[u32BoxsNum].u32MaxScoreIndex = u32MaxValueIndex + 1; //max class score index
                pstBox[u32BoxsNum].u32Mask = 0;

                u32BoxsNum++;
            }
        }
    }
    //quick_sort
    SAMPLE_SVP_NonRecursiveArgQuickSort(pstBox,0,u32BoxsNum-1,pstAssistStack);
    //Nms
    SAMPLE_SVP_NonMaxSuppression(pstBox,u32BoxsNum,f32NmsThresh,u32MaxBoxNum);
    //Get the result
    for (n = 0; (n < u32BoxsNum) && (u32BoxResultNum < u32MaxBoxNum); n++)
    {
        if (0 == pstBox[n].u32Mask)
        {
            pstBoxResult[u32BoxResultNum].f32Xmin = SVP_SAMPLE_MAX(pstBox[n].f32Xmin * u32SrcWidth, 0);
            pstBoxResult[u32BoxResultNum].f32Xmax = SVP_SAMPLE_MIN(pstBox[n].f32Xmax * u32SrcWidth, u32SrcWidth);
            pstBoxResult[u32BoxResultNum].f32Ymax = SVP_SAMPLE_MIN(pstBox[n].f32Ymax * u32SrcHeight, u32SrcHeight);
            pstBoxResult[u32BoxResultNum].f32Ymin = SVP_SAMPLE_MAX(pstBox[n].f32Ymin * u32SrcHeight, 0);
            pstBoxResult[u32BoxResultNum].f32ClsScore = pstBox[n].f32ClsScore;
            pstBoxResult[u32BoxResultNum].u32MaxScoreIndex = pstBox[n].u32MaxScoreIndex;

            u32BoxResultNum++;
        }
    }

    memcpy(pstBoxInfo->pstBbox, pstBoxResult, sizeof(SVP_SAMPLE_BOX_S)*u32OutBoxNum);
    pu32BoxNum[u32NumIndex] = u32OutBoxNum;

    SvpSampleDetectionPrint(pstBoxInfo, pu32BoxNum[u32NumIndex], strResultFolderDir, imgNameRecoder[u32NumIndex]);
    }

    //Free buffer
    /*SvpSampleMemFree(&stAssistBuf);
    SvpSampleMemFree(&stBoxsBuff);
    SvpSampleMemFree(&stBoxTmpBuf);
    SvpSampleMemFree(&stBoxsResultBuff);*/
    //free(pf32InputData);
   /* free(pstBoxResult);
    free(pstAssistStack);
    free(pstBox);
    free(stBoxTmpBuf);*/
}
