#ifndef _RFCN_INTERFACE_H_
#define _RFCN_INTERFACE_H_

#include "detectionCom.h"

using namespace std;

/**********************************parameter struct*************************************/

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

/**********************************software functions*************************************/

/***********************************************************************/
/* RFCN RPN calc function. calling after 1st seg                       */
/***********************************************************************/
HI_S32 SvpDetRfcnRpn(
    HI_S32 *ps32ClsScore,
    HI_S32 *ps32BBoxPred,
    NNIE_MODEL_INFO_S* pstModelInfo,
    HI_U32 u32NmsThresh,
    HI_U32 u32FilterThresh,
    HI_U32 u32NumBeforeNms,
    HI_U32 *pu32MemPool,
    HI_S32 *ps32ProposalResult,
    HI_U32 *pu32NumRois);

/************************************************************************/
/* RFCN get result calc function. calling after 3rd seg                 */
/************************************************************************/
HI_S32  SvpDetRfcnGetResult(
    SVP_SRC_MEM_INFO_S *pstSrc, HI_U32 u32FcScoreStride, HI_S32* ps32FcBbox, HI_U32 u32FcBboxStride,
    SVP_SRC_MEM_INFO_S *pstProposal, HI_U32 u32RoiCnt,
    SVP_DST_MEM_INFO_S *pstDstScore, SVP_DST_MEM_INFO_S *pstDstBbox,
    SVP_MEM_INFO_S *pstRoiOutCnt, NNIE_MODEL_INFO_S *pstModelInfo,
    HI_U32 u32ConfThresh, HI_U32 u32NmsThresh, SVP_MEM_INFO_S *pstMemPool,
    std::vector<RFCN_BoxesInfo> &vBoxesInfo, std::string& resultPath,
    SVP_SAMPLE_FILE_NAME_PAIR& imgNamePair);

/************************************************************************/
/* return RECN assist mem size                                          */
/************************************************************************/
HI_U32 SvpDetRfcnGetAssistMemSize(const stRFCNPara* para);

/************************************************************************/
/* SvpDetRfcnRpn function Alternative interface                         */
/************************************************************************/
HI_S32 rfcn_rpn(
    stRFCNPara* para,

    HI_S32* ps32ClsScore,
    HI_S32* ps32BBoxPred,
    HI_U32* pu32MemPool,
    HI_S32* bottom_rois,
    HI_U32& rois_num);

/************************************************************************/
/* SvpDetRfcnGetResult function Alternative interface                   */
/************************************************************************/
HI_S32 rfcn_detection_out(
    stRFCNPara* para,

    HI_S32* cls_prob_addr,                    // psroi loc output -> input
    HI_U32  cls_prob_stride,
    HI_S32* bbox_pred_addr,                   // psroi cls output -> input
    HI_U32  bbox_pred_stride,
    HI_S32* bottom_rois,                      // output coordinate
    HI_U32 rois_num,                          // output number
    HI_U32* result_mem_score,  HI_U32 result_mem_size_score,
    HI_U32* result_mem_bbox,   HI_U32 result_mem_size_bbox,
    HI_U32* result_mem_roiout, HI_U32 result_mem_size_roiout,
    HI_U32* pu32MemPool,
    std::vector<RFCN_BoxesInfo> &vBoxesInfo,
    std::string& resultPath,
    SVP_SAMPLE_FILE_NAME_PAIR& imgNamePair);

#endif /*_RFCN_INTERFACE_H_*/
