#ifndef __SVP_SAMPLE_WK_H__
#define __SVP_SAMPLE_WK_H__

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <iostream>
#include <utility>


#include "hi_type.h"
#include "hi_nnie.h"

#define STRIDE_ALIGN  (16)
#define HI_PI (3.1415926535897932384626433832795)

#define SVP_SAMPLE_MAX(a,b)    (((a) > (b)) ? (a) : (b))
#define SVP_SAMPLE_MIN(a,b)    (((a) < (b)) ? (a) : (b))

#define SVP_SAMPLE_ALIGN32(addr) ((((addr) + 32 - 1)/32)*32)
#define SVP_SAMPLE_ALIGN16(addr) ((((addr) + 16 - 1)/16)*16)

using namespace std;

//bbox四个参数，score，还有俩啥参数？
typedef struct hiSVP_SAMPLE_BOX_S
{
    HI_FLOAT f32Xmin;
    HI_FLOAT f32Xmax;
    HI_FLOAT f32Ymin;
    HI_FLOAT f32Ymax;
    HI_FLOAT f32ClsScore;
    HI_U32 u32MaxScoreIndex;//这是什么？
    HI_U32 u32Mask;//这是什么？
}SVP_SAMPLE_BOX_S;

//bbox信息结果，宽高，以及bbox块参数
typedef struct hiSVP_SAMPLE_BOX_RESULT_INFO_S
{
    HI_U32 u32OriImHeight;
    HI_U32 u32OriImWidth;
    SVP_SAMPLE_BOX_S* pstBbox;
}SVP_SAMPLE_BOX_RESULT_INFO_S;

//坐标的最大最小值
typedef struct hiSVP_SAMPLE_STACK_S {
    HI_S32     s32Min;      /*The minimum position coordinate */
    HI_S32     s32Max;      /*The maximum position coordinate */
} SVP_SAMPLE_STACK_S;

//模型参数，所有段共用的，某些段专用的
typedef struct hiSVP_WK_PARAM_RUNONCE_S
{
    // those below param is shared by all segments in one net.
    HI_U32 u32ModelBufSize;
    HI_U32 u32TmpBufSize;

    SVP_NNIE_MODEL_S stModel;
    SVP_MEM_INFO_S   stModelBuf;
    SVP_MEM_INFO_S   stTmpBuf;

    // those below param is owned by individual segment.
    SVP_MEM_INFO_S astTskBuf[SVP_NNIE_MAX_NET_SEG_NUM];
    HI_U32 au32TaskBufSize[SVP_NNIE_MAX_NET_SEG_NUM];

    SVP_SRC_BLOB_S stSrc[SVP_NNIE_MAX_INPUT_NUM];
    SVP_DST_BLOB_S stDst[SVP_NNIE_MAX_OUTPUT_NUM];
    SVP_BLOB_S stRPN[SVP_NNIE_MAX_OUTPUT_NUM];

    SVP_NNIE_FORWARD_CTRL_S stCtrl[SVP_NNIE_MAX_NET_SEG_NUM];
    SVP_NNIE_FORWARD_WITHBBOX_CTRL_S stBboxCtrl[SVP_NNIE_MAX_NET_SEG_NUM];
}SVP_WK_PARAM_RUNONECE_S;

//模型配置参数，名字，图片列表，最大输入数量，最大bbox数量，输出topN
typedef struct hiSVP_WK_CFG_S
{
    const HI_CHAR *pszModelName;
    const HI_CHAR *pszPicList;

    HI_U32 u32MaxInputNum;
    HI_U32 u32MaxBboxNum;

    HI_U32 u32TopN;
}SVP_WK_CFG_S;

//分类结果，分类id和得分值
typedef struct hiSVP_SAMPLE_CLF_RES_S
{
    HI_U32   u32ClassId;
    HI_U32   u32Confidence;
}SVP_SAMPLE_CLF_RES_S;

//第一个阶段？还是分割任务？
typedef struct hiSVP_NNIE_ONE_SEG_S
{
    HI_U32 u32TotalImgNum;
    FILE *fpSrc[SVP_NNIE_MAX_INPUT_NUM];
    FILE *fpLabel[SVP_NNIE_MAX_OUTPUT_NUM];

    HI_U32 u32ModelBufSize;
    HI_U32 u32TmpBufSize;

    SVP_NNIE_MODEL_S    stModel;
    SVP_MEM_INFO_S      stModelBuf;
    SVP_MEM_INFO_S      stTmpBuf;

    SVP_MEM_INFO_S      stTskBuf;
    HI_U32 u32TaskBufSize;

    SVP_SRC_BLOB_S astSrc[SVP_NNIE_MAX_INPUT_NUM];
    SVP_DST_BLOB_S astDst[SVP_NNIE_MAX_OUTPUT_NUM];

    SVP_NNIE_FORWARD_CTRL_S stCtrl;

    //memory needed by post-process of getting topN
    SVP_SAMPLE_CLF_RES_S *pstMaxClfIdScore;
    SVP_SAMPLE_CLF_RES_S *pastClfRes[SVP_NNIE_MAX_OUTPUT_NUM];
    HI_U32 au32ClfNum[SVP_NNIE_MAX_OUTPUT_NUM];
}SVP_NNIE_ONE_SEG_S;

//一阶段检测？如ssd，yolo等
typedef struct hiSVP_NNIE_ONE_SEG_DET_S
{
    HI_U32 u32TotalImgNum;
    FILE *fpSrc[SVP_NNIE_MAX_INPUT_NUM];
    FILE *fpLabel[SVP_NNIE_MAX_OUTPUT_NUM];

    HI_U32 u32ModelBufSize;
    HI_U32 u32TmpBufSize;

    SVP_NNIE_MODEL_S    stModel;
    SVP_MEM_INFO_S      stModelBuf;
    SVP_MEM_INFO_S      stTmpBuf;

    SVP_MEM_INFO_S      stTskBuf;
    HI_U32 u32TaskBufSize;

    SVP_SRC_BLOB_S astSrc[SVP_NNIE_MAX_INPUT_NUM];
    SVP_DST_BLOB_S astDst[SVP_NNIE_MAX_OUTPUT_NUM];

    SVP_NNIE_FORWARD_CTRL_S stCtrl;

    //memory needed by post-process of getting detection result
    HI_S32 *ps32ResultMem;    
}SVP_NNIE_ONE_SEG_DET_S;

//多阶段检测？如faster_rcnn
typedef struct hiSVP_NNIE_MULTI_SEG_S
{
    HI_U32 u32TotalImgNum;
    FILE *fpSrc[SVP_NNIE_MAX_INPUT_NUM];
    FILE *fpLabel[SVP_NNIE_MAX_OUTPUT_NUM];

    HI_U32 u32ModelBufSize;
    HI_U32 u32TmpBufSize;

    SVP_NNIE_MODEL_S    stModel;
    SVP_MEM_INFO_S      stModelBuf;
    SVP_MEM_INFO_S      stTmpBuf;

    // those below param is owned by individual segment.
    SVP_MEM_INFO_S      astTskBuf[SVP_NNIE_MAX_NET_SEG_NUM];
    HI_U32 au32TaskBufSize[SVP_NNIE_MAX_NET_SEG_NUM];

    SVP_SRC_BLOB_S astSrc[SVP_NNIE_MAX_INPUT_NUM];
    SVP_DST_BLOB_S astDst[SVP_NNIE_MAX_OUTPUT_NUM];
    SVP_BLOB_S stRPN[SVP_NNIE_MAX_OUTPUT_NUM];

    SVP_NNIE_FORWARD_CTRL_S astCtrl[SVP_NNIE_MAX_NET_SEG_NUM];
    SVP_NNIE_FORWARD_WITHBBOX_CTRL_S astBboxCtrl[SVP_NNIE_MAX_NET_SEG_NUM];
}SVP_NNIE_MULTI_SEG_S;

//nnie框架配置参数，模型名字，输入List，输出标签，最大输入，最大Bbox数，topN，标签数
typedef struct hiSVP_NNIE_CFG_S
{
    const HI_CHAR *pszModelName;
    const HI_CHAR *paszPicList[SVP_NNIE_MAX_INPUT_NUM];
    const HI_CHAR *paszLabel[SVP_NNIE_MAX_OUTPUT_NUM];

    HI_U32 u32MaxInputNum;
    HI_U32 u32MaxBboxNum;

    HI_U32 u32TopN;
    HI_BOOL bNeedLabel;
}SVP_NNIE_CFG_S;

//节点信息，层名称，所属段id，层id，输出id
typedef struct hiSVP_NNIE_NODE_INFO
{
    HI_CHAR layerName[SVP_NNIE_NODE_NAME_LEN];
    HI_U32 segID;
    HI_U32 layerID;
    HI_U32 dstIdx;
}SVP_NNIE_NODE_INFO;

//分类网络类型，lenet,alexnet,vggnet,googlenet,resnet50,squeezenet
typedef enum hiSVP_SAMPLE_WK_CLF_NET_TYPE_E
{
    SVP_SAMPLE_WK_CLF_NET_LENET         = 0x0,  /*LeNet*/
    SVP_SAMPLE_WK_CLF_NET_ALEXNET       = 0x1,  /*Alexnet*/
    SVP_SAMPLE_WK_CLF_NET_VGG16         = 0x2,  /*Vgg16*/
    SVP_SAMPLE_WK_CLF_NET_GOOGLENET     = 0x3,  /*Googlenet*/
    SVP_SAMPLE_WK_CLF_NET_RESNET50      = 0x4,  /*Resnet50*/
    SVP_SAMPLE_WK_CLF_NET_SQUEEZENET    = 0x5,  /*Squeezenet*/

    SVP_SAMPLE_WK_CLF_NET_TYPE_BUTT
}SVP_SAMPLE_WK_CLF_NET_TYPE_E;

//检测网络类型，yolov1,yolov2,ssd
typedef enum hiSVP_SAMPLE_WK_DETECT_NET_TYPE_E
{
    SVP_SAMPLE_WK_DETECT_NET_YOLOV1   =  0x0,  /*Yolov1*/
    SVP_SAMPLE_WK_DETECT_NET_YOLOV2   =  0x1,  /*Yolov2*/
    SVP_SAMPLE_WK_DETECT_NET_SSD      =  0x2,  /*Ssd*/

    SVP_SAMPLE_WK_DETECT_NET_TYPE_BUTT
}SVP_SAMPLE_WK_DETECT_NET_TYPE_E;

//faster rcnn模型类型，alexnet,vgg16,resnet18,resnet34,pvanet,doubleroi
typedef enum hiSVP_SAMPLE_WK_DETECT_NET_FASTER_RCNN_TYPE_E
{
    SVP_SAMPLE_WK_DETECT_NET_FASTER_RCNN_ALEX   =  0x0,  /*fasterrcnn_alexnet*/
    SVP_SAMPLE_WK_DETECT_NET_FASTER_RCNN_VGG16,          /*fasterrcnn_vgg16*/
    SVP_SAMPLE_WK_DETECT_NET_FASTER_RCNN_RES18,          /*fasterrcnn_resnet18*/
    SVP_SAMPLE_WK_DETECT_NET_FASTER_RCNN_RES34,          /*fasterrcnn_resnet34*/
    SVP_SAMPLE_WK_DETECT_NET_FASTER_RCNN_PVANET,         /*fasterrcnn_pvanet*/
    SVP_SAMPLE_WK_DETECT_NET_FASTER_RCNN_DOUBLE_ROI,     /*fasterrcnn_double_roi*/

    SVP_SAMPLE_WK_DETECT_NET_FASTER_RCNN_TYPE_BUTT
}SVP_SAMPLE_WK_DETECT_NET_FASTER_RCNN_TYPE_E;

//rfcn模型类型，resnet50
typedef enum hiSVP_SAMPLE_WK_DETECT_NET_RFCN_TYPE_E
{
    SVP_SAMPLE_WK_DETECT_NET_RFCN_RES50   =  0x0,

    SVP_SAMPLE_WK_DETECT_NET_RFCN_TYPE_BUTT
}SVP_SAMPLE_WK_DETECT_NET_RFCN_TYPE_E;

//lstm运行时参数
typedef struct _LSTMRunTimeCtx
{
    HI_U32 *pu32Seqs;
    HI_U32 u32SeqNr;
    HI_U32 u32MaxT;
    HI_U32 u32TotalT;
    HI_U8 u8ExposeHid;
    HI_U8 u8WithStatic;
} SVP_SAMPLE_LSTMRunTimeCtx;

//分类和一阶段的检测
/*classification with input images and labels, print the top-N result */
HI_S32 SvpSampleCnnClassification(const HI_CHAR *pszModelName, const HI_CHAR *paszPicList[], const HI_CHAR *paszLabel[], HI_S32 s32Cnt=1);
HI_S32 SvpSampleCnnClassificationForword(SVP_NNIE_ONE_SEG_S *pstClfParam, SVP_NNIE_CFG_S *pstClfCfg);

HI_S32 SvpSampleCnnDetectionOneSeg(const HI_CHAR *pszModelName, const HI_CHAR *paszPlicList[], const HI_U8 netType, HI_S32 s32Cnt=1);

#define LSTM_UT_EXPOSE_HID 2
#define LSTM_UT_WITH_STATIC 1
void SvpSampleCreateLSTMCtx(SVP_SAMPLE_LSTMRunTimeCtx *pstCtx, HI_U32 u32SentenceNr, HI_U32 u32BaseFrameNr,
    HI_U8 u8ExposeHid, HI_U8 u8WithStatic);
void SvpSampleDestoryLSTMCtx(SVP_SAMPLE_LSTMRunTimeCtx *pstCtx);
HI_BOOL SvpSampleWkLSTM(const HI_CHAR *pszModelName, const HI_CHAR *pszPicList[], HI_U32 u32PicListNum,
    HI_U32 *pu32SrcAlign, HI_U32 *pu32DstAlign, SVP_SAMPLE_LSTMRunTimeCtx *pstCtx = NULL);

//二阶段的检测，faster_rcnn入口函数
HI_S32 SvpSampleWKFasterRCNNRun(const HI_CHAR *pszModel, const HI_CHAR *pszPicList[],
    HI_U8 netType, HI_U32 *pu32DstAlign, HI_S32 s32Cnt=1);

//rfcn入口函数
HI_S32 SvpSampleWKRFCNRun(const HI_CHAR *pszModel, const HI_CHAR *pszPicList[],
    HI_U32 *pu32DstAlign,  HI_S32 s32Cnt=1);

//lstm的初始化和注销函数
HI_S32 SvpSampleLSTMInit(SVP_NNIE_CFG_S *pstClfCfg, SVP_NNIE_ONE_SEG_S *pstComParam, SVP_SAMPLE_LSTMRunTimeCtx *pstCtx);
HI_S32 SvpSampleLSTMDeinit(SVP_NNIE_ONE_SEG_S *pstComParam);

/*Segmentation*/
HI_S32 SvpSampleSegnet(const HI_CHAR *pszModelName, const HI_CHAR *paszPicList[], HI_S32 s32Cnt=1);

/* one seg net mem intial */
HI_S32 SvpSampleOneSegCnnInit(SVP_NNIE_CFG_S *pstClfCfg, SVP_NNIE_ONE_SEG_S *pstComParam);
/* Unload model, and free memory */
void SvpSampleOneSegCnnDeinit(SVP_NNIE_ONE_SEG_S *pstComParam);

HI_S32 SvpSampleOneSegDetCnnInit(SVP_NNIE_CFG_S *pstClfCfg, SVP_NNIE_ONE_SEG_DET_S *pstComfParam,const HI_U8 netType);

void SvpSampleOneSegDetCnnDeinit(SVP_NNIE_ONE_SEG_DET_S *pstComParam);

HI_S32 SvpSampleMultiSegCnnInit(SVP_NNIE_CFG_S *pstComCfg, SVP_NNIE_MULTI_SEG_S *pstComParam,
    HI_U32 *pu32SrcAlign = NULL, HI_U32 *pu32DstAlign = NULL);

void SvpSampleMultiSegCnnDeinit(SVP_NNIE_MULTI_SEG_S *pstComParam);


#define SVP_NNIE_MAX_RATIO_ANCHOR_NUM (32) /*NNIE max ratio anchor num*/
#define SVP_NNIE_MAX_SCALE_ANCHOR_NUM (32) /*NNIE max scale anchor num*/

//基于anchor的信息
typedef struct hiSVP_SAMPLE_BASE_ANCHOR_INFO_S
{
    HI_U32 u32NumRatioAnchors;
    HI_U32 u32NumScaleAnchors;
    HI_U32 au32Scales[SVP_NNIE_MAX_RATIO_ANCHOR_NUM];
    HI_U32 au32Ratios[SVP_NNIE_MAX_SCALE_ANCHOR_NUM];
}SVP_SAMPLE_BASE_ANCHOR_INFO_S;

//基于anchor的信息初始化
/* base anchor information initialize */
HI_S32 SvpSampleAnchorInfoInit(SVP_SAMPLE_WK_DETECT_NET_FASTER_RCNN_TYPE_E netType,
    SVP_SAMPLE_BASE_ANCHOR_INFO_S* baseAnchorInfo);

//grid数量
#define SVP_SAMPLE_YOLOV2_GRIDNUM        (13)
//channel数量
#define SVP_SAMPLE_YOLOV2_CHANNLENUM     (50)
//参数量
#define SVP_SAMPLE_YOLOV2_PARAMNUM       (10)
//box数量
#define SVP_SAMPLE_YOLOV2_BOXNUM         (5)
//类别数量
#define SVP_SAMPLE_YOLOV2_CLASSNUM       (5)
//box总数，13*13*5
#define SVP_SAMPLE_YOLOV2_BOXTOTLENUM    (SVP_SAMPLE_YOLOV2_GRIDNUM * SVP_SAMPLE_YOLOV2_GRIDNUM * SVP_SAMPLE_YOLOV2_BOXNUM)
//最大box数量
#define SVP_SAMPLE_YOLOV2_MAX_BOX_NUM    (10)
//宽
#define SVP_SAMPLE_YOLOV2_WIDTH          (7)

//ssd参数，类别个数，topk，box总数
/************************/
typedef struct hiResult_SSD_Para_S
{
    HI_U32 u32NumClasses;
    HI_U32 u32KeepTopK;
    HI_U32 u32TotlBoxNum;

}Result_SSD_Para_S;
/************************/

//分类模型
/*Classificacion*/
void SvpSampleCnnClfLenet();
void SvpSampleCnnClfAlexnet();
void SvpSampleCnnClfVgg16();
void SvpSampleCnnClfGooglenet();
void SvpSampleCnnClfResnet50();
void SvpSampleCnnClfSqueezenet();

//检测模型
/*Detection*/
void SvpSampleRoiDetFasterRCNNAlexnet();
void SvpSampleRoiDetFasterRCNNVGG16();
void SvpSampleRoiDetFasterRCNNResnet18();
void SvpSampleRoiDetFasterRCNNResnet34();
void SvpSampleRoiDetFasterRCNNPvanet();
void SvpSampleRoiDetFasterRCNNDoubleRoi();
void SvpSampleRoiDetRFCNResnet50();
void SvpSampleCnnDetYoloV1();
void SvpSampleCnnDetYoloV2();
void SvpSampleCnnDetSSD();

//分割模型
/*Segmentation*/
void SvpSampleCnnFcnSegnet();

/*LSTM*/
void SvpSampleRecurrentLSTMFC();
void SvpSampleRecurrentLSTMRelu();

//文件名称对，文件名+后缀
/*SVP_SAMPLE_FILE_NAME_PAIR first:  basic filename, second: filename suffix*/
typedef pair<string, string> SVP_SAMPLE_FILE_NAME_PAIR;

//获取或者打印检测结果，yolov1，yolov2，ssd
/*Detection get result*/
void SvpSampleWkYoloV1GetResult(SVP_BLOB_S *pstDstBlob, SVP_SAMPLE_BOX_RESULT_INFO_S *pstBoxes,
    HI_U32 *pu32BoxNum, string& strResultFolderDir, vector<SVP_SAMPLE_FILE_NAME_PAIR>& imgNameRecoder);

void SvpSampleWkYoloV2GetResult(SVP_BLOB_S *pstDstBlob, HI_S32 *ps32ResultMem,
    SVP_SAMPLE_BOX_RESULT_INFO_S *pstBoxes, HI_U32 *pu32BoxNum, string& strResultFolderDir,
    vector<SVP_SAMPLE_FILE_NAME_PAIR>& imgNameRecoder);

void SvpSampleWkSSDGetResult(SVP_NNIE_ONE_SEG_DET_S *pstDetParam, HI_VOID *pstSSDParam,
    SVP_SAMPLE_BOX_RESULT_INFO_S *pstBoxes, HI_U32 *pu32BoxNum, string& strResultFolderDir,
    vector<SVP_SAMPLE_FILE_NAME_PAIR>& imgNameRecoder);

void SvpSampleDetectionPrint(const SVP_SAMPLE_BOX_RESULT_INFO_S *pstResultBoxesInfo,
    HI_U32 u32BoxNum, string& strResultFolderDir, SVP_SAMPLE_FILE_NAME_PAIR& imgNamePair);

#endif //__SVP_SAMPLE_WK_H__
