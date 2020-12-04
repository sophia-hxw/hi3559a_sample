/*
*** 注释标签说明：
*** QUES: 没看懂的问题，待解释？
*** TODO: 懂抽象层面含义，需要补充细节
*/
#include "SvpSampleWk.h"
#include "SvpSampleCom.h"

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

//定义模型参数
#define YOLOV1_IMG_WIDTH   (448)
#define YOLOV1_IMG_HEIGHT  (448)
#define YOLOV1_BBOX_CNT    (98)
#define YOLOV1_CLASS_CNT   (20)
#define YOLOV1_THRESHOLD   (0.2)
#define YOLOV1_IOU         (0.5)

#define YOLOV1_GRID_NUM    (7)
#define YOLOV1_CHANNEL_NUM (30)
#define YOLOV1_BOX_NUM     (2)

#ifndef SVP_WK_COORDI_NUM
#define SVP_WK_COORDI_NUM  (4)
#endif

//QUES: 这个参数是什么？
#ifndef SVP_WK_QUANT_BASE
#define SVP_WK_QUANT_BASE  (0x1000)//十六进制数字，等于十进制的4096
#endif

//类别label
const HI_CHAR classes[20][20] =
{
    "aeroplane",   "bicycle", "bird",  "boat",      "bottle",
    "bus",         "car",     "cat",   "chair",     "cow",
    "diningtable", "dog",     "horse", "motorbike", "person",
    "pottedplant", "sheep",   "sofa",  "train",     "tvmonitor"
};

//定义结果描述结构体
typedef struct st_yolov1_score
{
    HI_U32 idx;
    HI_FLOAT value;
}yolov1_score;

//位置描述参数
typedef struct st_position
{
    HI_FLOAT x;
    HI_FLOAT y;
    HI_FLOAT w;
    HI_FLOAT h;
}position;

//定义两个float变量在板子上的相等
HI_U32 FLOAT_EQUAL(HI_FLOAT a, HI_FLOAT b)
{
    return fabs(a - b) < 0.000001;
}

//计算两个bbox的iou
HI_FLOAT YoloV1CalIou(position *bbox, HI_U32 bb1, HI_U32 bb2)
{
    HI_FLOAT tb, lr;
    HI_FLOAT intersection;

    tb = SVP_SAMPLE_MIN(bbox[bb1].x + 0.5f*bbox[bb1].w, bbox[bb2].x + 0.5f*bbox[bb2].w)
    - SVP_SAMPLE_MAX(bbox[bb1].x - 0.5f*bbox[bb1].w, bbox[bb2].x - 0.5f*bbox[bb2].w);
    lr = SVP_SAMPLE_MIN(bbox[bb1].y + 0.5f*bbox[bb1].h, bbox[bb2].y + 0.5f*bbox[bb2].h)
    - SVP_SAMPLE_MAX(bbox[bb1].y - 0.5f*bbox[bb1].h, bbox[bb2].y - 0.5f*bbox[bb2].h);

    if (tb < 0 || lr < 0)
        intersection = 0;
    else intersection = tb*lr;

    return intersection / (bbox[bb1].w*bbox[bb1].h + bbox[bb2].w*bbox[bb2].h - intersection);
}

//比较两个score，QUES: 形式参数的类型对么？
HI_S32 cmp(const void *a, const void *b)
{
    return ((yolov1_score*)a)->value < ((yolov1_score*)b)->value;
}

//按score给得分数组排序，参数：得分值score数组指针，序列结果指针
void GetSortedIdx(HI_FLOAT *array, HI_U32 *idx)
{
    yolov1_score tmp[YOLOV1_BBOX_CNT];
    //YOLOV1_BBOX_CNT（最大=98）个检测结构排序？
    HI_U32 i = 0;

    for (i = 0; i < YOLOV1_BBOX_CNT; ++i)
    {
        tmp[i].idx = i;
        tmp[i].value = array[i];
    }

    //TODO: 理解qsort的含义
    qsort(tmp, YOLOV1_BBOX_CNT, sizeof(yolov1_score), cmp);

    for (i = 0; i < YOLOV1_BBOX_CNT; ++i)
        idx[i] = tmp[i].idx;
}

//yolov1的NMS，参数：得分值数组，对应的bbox
void YoloV1NMS(HI_FLOAT *array, position *bbox)
{
    HI_U32 result[YOLOV1_BBOX_CNT];//排序结果
    HI_U32 i, j;
    HI_U32 idx_i, idx_j;

    for (i = 0; i < YOLOV1_BBOX_CNT; ++i)
    {
        if (array[i] < YOLOV1_THRESHOLD)//先截断了一个较低的阈值，这里是0.2
            array[i] = 0.0;
    }

    //排序
    GetSortedIdx(array, result);

    //整体思路，输入的是两个数组，得分和框，nms滤掉的框直接得分置0，剩下的非0就是需要的
    for (i = 0; i < YOLOV1_BBOX_CNT; ++i)//所有框
    {
        idx_i = result[i];//取得分值最大那个

        if (FLOAT_EQUAL(array[idx_i], 0.0)) continue;
        for (j = i + 1; j < YOLOV1_BBOX_CNT; ++j)//剩下的框
        {
            idx_j = result[j];

            if (FLOAT_EQUAL(array[idx_j], 0.0)) continue;

            if (YoloV1CalIou(bbox, idx_i, idx_j) > YOLOV1_IOU)
                array[idx_j] = 0.0;
        }
    }
}

//位置参数转换，[x,y,w,h]->[xmin, xmax, ymin, ymax]
void ConvertPosition(position bbox, SVP_SAMPLE_BOX_S *result)
{
    HI_FLOAT xMin, yMin, xMax, yMax;
    xMin = bbox.x - 0.5f * bbox.w;
    yMin = bbox.y - 0.5f * bbox.h;
    xMax = bbox.x + 0.5f * bbox.w;
    yMax = bbox.y + 0.5f * bbox.h;

    xMin = xMin > 0 ? xMin : 0;
    yMin = yMin > 0 ? yMin : 0;
    xMax = xMax > YOLOV1_IMG_WIDTH ? YOLOV1_IMG_WIDTH : xMax;
    yMax = yMax > YOLOV1_IMG_HEIGHT ? YOLOV1_IMG_HEIGHT : yMax;

    result->f32Xmin = xMin;
    result->f32Ymin = yMin;
    result->f32Xmax = xMax;
    result->f32Ymax = yMax;
}

//yolov1的检测，参数：二维得分数组，模型结果[x,y,w,h]位置指针
//后续分析结构bbox[x1,x2,y1,y2]指针，返回检测的个数
HI_U32 YoloV1Detect(HI_FLOAT af32Score[][YOLOV1_BBOX_CNT], position *bbox, SVP_SAMPLE_BOX_S *pstBoxesResult)
{
    HI_U32 i, j;
    HI_U32 ans_idx = 0;
    HI_FLOAT maxScore = 0.0;
    HI_U32 U32_MAX = 0xFFFFFFFF;//十六进制数，十进制是4294967295，二进制是32个1
    HI_U32 idx = U32_MAX;

    for (i = 0; i < YOLOV1_CLASS_CNT; ++i)//每个类别先做nms
        YoloV1NMS(af32Score[i], bbox);

    for (i = 0; i < YOLOV1_BBOX_CNT; ++i)//i是框的索引
    {
        maxScore = 0.0;
        idx = -1;

        //对每个框，选择score最大的那个类别作为该框的类别
        for (j = 0; j < YOLOV1_CLASS_CNT; ++j)//j是类别的索引
        {
            if (af32Score[j][i] > maxScore)
            {
                maxScore = af32Score[j][i];
                idx = j;
            }
        }

        if (idx != U32_MAX)
        {
            pstBoxesResult[ans_idx].u32MaxScoreIndex = idx;
            //得分score最大的类别class索引
            pstBoxesResult[ans_idx].f32ClsScore = maxScore;
            //分类得分值
            ConvertPosition(bbox[i], &pstBoxesResult[ans_idx]);
            ++ans_idx;
        }
    }

    return ans_idx;
}

//获取yolov1的检测结果，参数：输出blob，结果bbox信息结果，？，结果文件路径，“文件名+后缀”数组
void SvpSampleWkYoloV1GetResult(SVP_BLOB_S *pstDstBlob, SVP_SAMPLE_BOX_RESULT_INFO_S *pstBoxInfo, HI_U32 *pu32BoxNum,
    string& strResultFolderDir, vector<SVP_SAMPLE_FILE_NAME_PAIR>& imgNameRecoder)
{
    HI_FLOAT af32Scores[YOLOV1_CLASS_CNT][YOLOV1_BBOX_CNT];//YOLOV1_CLASS_CNT最大20，YOLOV1_BBOX_CNT最大98
    HI_FLOAT *pf32ClassProbs=NULL;
    HI_FLOAT *pf32Confs=NULL;//confidence?
    HI_FLOAT *pf32boxes=NULL;
    HI_FLOAT f32InputData[YOLOV1_CHANNEL_NUM * YOLOV1_GRID_NUM * YOLOV1_GRID_NUM];//30*7*7

    HI_U32 i, j, k, idx = 0;

    pf32ClassProbs = f32InputData;
    pf32Confs = pf32ClassProbs + YOLOV1_CLASS_CNT * YOLOV1_GRID_NUM * YOLOV1_GRID_NUM;//20*7*7
    pf32boxes = pf32Confs + YOLOV1_BBOX_CNT;//98

    HI_U32 u32FrameStride = pstDstBlob->u32Stride * pstDstBlob->unShape.stWhc.u32Chn * pstDstBlob->unShape.stWhc.u32Height;
    //Stride*channel*height

    for (HI_U32 u32NumIndex = 0; u32NumIndex < pstDstBlob->u32Num; u32NumIndex++)//caffe里nchw的n
    {

        for (i = 0; i < YOLOV1_CHANNEL_NUM * YOLOV1_GRID_NUM * YOLOV1_GRID_NUM; ++i)//30*7*7
        {
            f32InputData[i] = ((HI_S32*)((HI_U8*)pstDstBlob->u64VirAddr + u32NumIndex * u32FrameStride))[i] * 1.0f / SVP_WK_QUANT_BASE;
        }

        for (i = 0; i < YOLOV1_GRID_NUM * YOLOV1_GRID_NUM; ++i)//7*7
        {
            for (j = 0; j < YOLOV1_BOX_NUM; ++j)//bbox数量，2
            {
                for (k = 0; k < YOLOV1_CLASS_CNT; ++k)//类别，20
                {
                    af32Scores[k][idx] = (*(pf32ClassProbs + i * YOLOV1_CLASS_CNT + k))* (*(pf32Confs + i * YOLOV1_BOX_NUM + j));
                }
                ++idx;
            }
        }
        idx = 0;

        for (i = 0; i < YOLOV1_GRID_NUM * YOLOV1_GRID_NUM; ++i)//7*7
        {
            for (j = 0; j < YOLOV1_BOX_NUM; ++j)//bbox数量，2
            {
                pf32boxes[(i * 2 + j) * 4 + 0] = (pf32boxes[(i * 2 + j) * 4 + 0] + i % YOLOV1_GRID_NUM) / YOLOV1_GRID_NUM * YOLOV1_IMG_WIDTH;//x
                pf32boxes[(i * 2 + j) * 4 + 1] = (pf32boxes[(i * 2 + j) * 4 + 1] + i / YOLOV1_GRID_NUM) / YOLOV1_GRID_NUM * YOLOV1_IMG_HEIGHT;//y
                pf32boxes[(i * 2 + j) * 4 + 2] = pf32boxes[(i * 2 + j) * 4 + 2] * pf32boxes[(i * 2 + j) * 4 + 2] * YOLOV1_IMG_WIDTH;//w
                pf32boxes[(i * 2 + j) * 4 + 3] = pf32boxes[(i * 2 + j) * 4 + 3] * pf32boxes[(i * 2 + j) * 4 + 3] * YOLOV1_IMG_HEIGHT;//h
            }
        }

        pu32BoxNum[u32NumIndex] = YoloV1Detect(af32Scores, (position*)(pf32boxes), pstBoxInfo->pstBbox);
        SvpSampleDetectionPrint(pstBoxInfo, pu32BoxNum[u32NumIndex], strResultFolderDir, imgNameRecoder[u32NumIndex]);
    }
}
