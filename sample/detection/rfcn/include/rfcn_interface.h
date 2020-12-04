#ifndef _RFCN_INTERFACE_H_
#define _RFCN_INTERFACE_H_

#include <iostream>
#include <cfloat>
#include <algorithm>
#include <cstdlib>
#include <time.h>
#include <string>
#include <fstream>
#include <vector>
#include "hi_type.h"
#include "genRoiInfo.h"
#include "detectionCom.h"

using namespace std;

#define RFCN_DEBUG (1)

typedef enum t_pooling_type
{
    PoolingParameter_PoolMethod_MAX = 0,
    PoolingParameter_PoolMethod_AVE,
    PoolingParameter_PoolMethod_STOCHASTIC,
}pooling_type;

#define TRANS_U8_TO_ADDR  (unsigned long long)(unsigned long)
#define TRANS_ADDR_TO_U32  (unsigned int*)(unsigned long)
#define TRANS_ADDR_TO_U8  (unsigned char*)(unsigned long)

struct stRFCNPara
{
//---------- parameters for PriorBox ---------
    HI_U32 u32NumBeforeNms;     // 6000

    NNIE_MODEL_INFO_S model_info;

    HI_U32 data_size;         // input

    HI_U32 u32NmsThresh;       // 0.7
    HI_U32 u32ValidNmsThresh;  // 0.3
    HI_U32 u32FilterThresh;    // 0
    HI_U32 u32ConfThresh;      // 0.3
};

typedef struct tagRFCN_BoxesInfo
{
    HI_U32 u32XMin;
    HI_U32 u32YMin;
    HI_U32 u32XMax;
    HI_U32 u32YMax;
    HI_U32 u32Class;
    HI_FLOAT fScore;
} RFCN_BoxesInfo;

HI_S32 HI_MPI_SVP_NNIE_WK_CNN_FASTER_RPN_Ref(
    HI_S32 *ps32Src,
    HI_S32 *ps32BBoxPred,
    NNIE_MODEL_INFO_S* pstModelInfo,
    HI_U32 u32NmsThresh,
    HI_U32 u32FilterThresh,
    HI_U32 u32NumBeforeNms,
    HI_U32 *pu32MemPool,
    HI_S32 *ps32ProposalResult,
    HI_U32 *pu32NumRois
    );

HI_S32  HI_MPI_SVP_NNIE_WK_RFCN_GetResult(
    SVP_SRC_MEM_INFO_S *pstSrc, HI_U32 u32FcScoreStride, HI_S32* ps32FcBbox, HI_U32 u32FcBboxStride,
    SVP_SRC_MEM_INFO_S *pstProposal, HI_U32 u32RoiCnt,
    SVP_DST_MEM_INFO_S *pstDstScore, SVP_DST_MEM_INFO_S *pstDstBbox,
    SVP_MEM_INFO_S *pstRoiOutCnt, NNIE_MODEL_INFO_S *pstModelInfo,
    HI_U32 u32ConfThresh, HI_U32 u32NmsThresh, SVP_MEM_INFO_S *pstMemPool,
    std::vector<RFCN_BoxesInfo> &vBoxesInfo, std::string& resultPath,
    SVP_SAMPLE_FILE_NAME_PAIR& imgNamePair);

HI_S32 rfcn_rpn(
    stRFCNPara* para,

    HI_S32* report_0_data,                    //conv4
    HI_S32* report_1_data,                    //conv5
    HI_U32* pu32MemPool,                      //assist mem 10000
    HI_S32* bottom_rois,
    HI_U32& rois_num
    );

HI_S32 rfcn_detection_out(
    stRFCNPara* para,

    HI_U32* bottom_data1,                     // psroi loc output -> input
    HI_U32 score_stride,
    HI_S32* bottom_data_2,                    // psroi cls output -> input
    HI_U32 bbox_stride,
    HI_S32* bottom_rois,                      // output coordinate
    HI_U32 rois_num,                          // output number
    HI_U32* result1, HI_U32* length1,
    HI_U32* result2, HI_U32* length2,
    HI_U32* result3, HI_U32* length3,
    HI_U32* pu32MemPool,                       //assist mem 10000
    std::vector<RFCN_BoxesInfo> &vBoxesInfo,
    std::string& resultPath,
    SVP_SAMPLE_FILE_NAME_PAIR& imgNamePair);

HI_U32 GetRFCNAssistMemSize(stRFCNPara* para);


#endif /*_RFCN_INTERFACE_H_*/
