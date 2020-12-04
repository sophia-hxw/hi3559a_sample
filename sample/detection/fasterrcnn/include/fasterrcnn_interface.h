#ifndef _FASTERRCNN_INTERFACE_H_
#define _FASTERRCNN_INTERFACE_H_


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hi_type.h"
#include "hi_nnie.h"
#include "detectionCom.h"

/**********************************parameter struct*************************************/
typedef struct hiFaster_RCNN_Para
{
    //----------RPN PARAMETER----------
    HI_U32 u32NumRatioAnchors;
    HI_U32 u32NumScaleAnchors;
    HI_U32 au32Scales[32];
    HI_U32 au32Ratios[32];

    HI_U32 u32OriImHeight;
    HI_U32 u32OriImWidth;

    HI_U32 au32ConvHeight[3];
    HI_U32 au32ConvWidth[3];
    HI_U32 au32ConvChannel[3];
    HI_U32 au32ConvStride[3];

    HI_U32 u32MaxRois;
    HI_U32 u32MinSize;
    HI_U32 u32SpatialScale;

    HI_U32 u32NmsThresh;
    HI_U32 u32FilterThresh;
    HI_U32 u32NumBeforeNms;
    HI_U32 u32NumRois;
    NNIE_MODEL_INFO_S model_info;

    //----------GET RESULT PARAMETER----------
    HI_U32 u32ConfThresh;
    HI_U32 u32ValidNmsThresh;
    HI_U32 u32ClassNum;
}Faster_RCNN_Para;

/**********************************software functions*************************************/
HI_S32 HI_MPI_SVP_NNIE_WK_CNN_FASTER_RPN_Ref1(
        HI_S32 **ps32Src,
        HI_U32 u32NumRatioAnchors,
        HI_U32 u32NumScaleAnchors,
        HI_U32* pu32Scales,
        HI_U32* pu32Ratios,
        HI_U32 u32OriImHeight,
        HI_U32 u32OriImWidth,

        // 3 conv input parameters
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
        HI_U32 *pu32MemPool,
        HI_S32 *ps32ProposalResult,
        HI_U32 *pu32NumRois
);

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
        HI_U32  u32NmsThresh,
        HI_U32  u32MaxRois,
        HI_U32  u32ClassNum,
        HI_U32  u32OriImWidth,
        HI_U32  u32OriImHeight,
        HI_S32* pstMemPool);  // assist mem

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
);

HI_U32 malloc_rpn_assist_mem_size(Faster_RCNN_Para* para);
HI_U32 malloc_get_result_assist_mem_size(Faster_RCNN_Para* para);
HI_S32 write_result(Faster_RCNN_Para* para, HI_S32* pstDstScore, HI_S32* pstDstBbox, HI_S32* pstRoiOutCnt,
    std::string& resultPath, SVP_SAMPLE_FILE_NAME_PAIR& imgNamePair);
HI_S32 Print_Result1(HI_CHAR *filename, HI_CHAR *buff, HI_U32 data_len, HI_U32 data_with);

#endif /* _FASTERRCNN_INTERFACE_H_ */
