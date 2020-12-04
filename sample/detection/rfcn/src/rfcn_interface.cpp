#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <math.h>
#include <vector>
#include <string.h>

#include "rfcn_interface.h"
#include "detectionCom.h"

using namespace std;

HI_S32 rfcn_rpn(
    stRFCNPara* para,

    HI_S32* report_0_data,
    HI_S32* report_1_data,
    HI_U32* pu32MemPool,
    HI_S32* bottom_rois,
    HI_U32& rois_num)
{
    HI_U32 u32NmsThresh = para->u32NmsThresh;
    HI_U32 u32FilterThresh = para->u32FilterThresh;
    HI_S32 u32NumBeforeNms = para->u32NumBeforeNms;

    NNIE_MODEL_INFO_S  model_info = para->model_info;

    return HI_MPI_SVP_NNIE_WK_CNN_FASTER_RPN_Ref(
        report_0_data,
        report_1_data,
        &model_info,
        u32NmsThresh,
        u32FilterThresh,
        u32NumBeforeNms,
        pu32MemPool,
        bottom_rois,
        &rois_num);
}

HI_S32 rfcn_detection_out(
    stRFCNPara* para,

    HI_U32* bottom_data_1,                      //psroi cls output -> input
    HI_U32 score_stride,
    HI_S32* bottom_data_2,                      //psroi loc output -> input
    HI_U32 bbox_stride,
    HI_S32* bottom_rois,
    HI_U32 rois_num,
    HI_U32* result1, HI_U32* length1,
    HI_U32* result2, HI_U32* length2,
    HI_U32* result3, HI_U32* length3,
    HI_U32* pu32MemPool, std::vector<RFCN_BoxesInfo> &vBoxesInfo,
    std::string& resultPath, SVP_SAMPLE_FILE_NAME_PAIR& imgNamePair)
{
    //============================================
    HI_U32 u32ConfThresh = para->u32ConfThresh;
    HI_U32 u32ValidNmsThresh = para->u32ValidNmsThresh;

    NNIE_MODEL_INFO_S model_info = para->model_info;

    //get result
    SVP_SRC_MEM_INFO_S pstSrc;
    pstSrc.u64VirAddr = (HI_U64)bottom_data_1;
    pstSrc.u32Size = para->data_size * sizeof(HI_U32);

    SVP_SRC_MEM_INFO_S pstProp;
    pstProp.u64VirAddr = (HI_U64)bottom_rois;
    pstProp.u32Size = model_info.u32MaxRoiFrameCnt * SVP_WK_COORDI_NUM * sizeof(HI_U32);

    SVP_SRC_MEM_INFO_S pstDstScore;
    *length1 = model_info.u32MaxRoiFrameCnt * model_info.u32ClassSize;
    pstDstScore.u64VirAddr = (HI_U64)result1;
    pstDstScore.u32Size = (*length1) * sizeof(HI_U32);

    SVP_SRC_MEM_INFO_S pstDstBbox;
    *length2 = model_info.u32MaxRoiFrameCnt * model_info.u32ClassSize * SVP_WK_COORDI_NUM;
    pstDstBbox.u64VirAddr = (HI_U64)result2;
    pstDstBbox.u32Size = (*length2) * sizeof(HI_U32);

    SVP_SRC_MEM_INFO_S pstRoiOutCnt;
    *length3 = model_info.u32ClassSize;
    pstRoiOutCnt.u64VirAddr = (HI_U64)result3;
    pstRoiOutCnt.u32Size = (*length3) * sizeof(HI_U32);

    SVP_SRC_MEM_INFO_S pstMemPool;
    pstMemPool.u64VirAddr = (HI_U64)pu32MemPool;
    pstMemPool.u32Size =  (*length1) * sizeof(HI_U32);

    HI_U32 u32FcScoreStride = score_stride;
    HI_U32 u32FcBboxStride = bbox_stride;


    HI_MPI_SVP_NNIE_WK_RFCN_GetResult(
        &pstSrc, u32FcScoreStride, bottom_data_2,
        u32FcBboxStride, &pstProp, rois_num,
        &pstDstScore, &pstDstBbox,
        &pstRoiOutCnt, &model_info,
        u32ConfThresh, u32ValidNmsThresh, &pstMemPool,
        vBoxesInfo, resultPath, imgNamePair);

#if RFCN_DEBUG
        FILE* file = NULL;
        string file_path;

        /******************* write output_score **********************/
        file_path = resultPath + imgNamePair.first + "_output_score.txt";
        file = fopen(file_path.c_str(), "w");
        SVP_CHECK(NULL != file, HI_FAILURE);

        for (HI_U32 i = 0; i < model_info.u32MaxRoiFrameCnt * model_info.u32ClassSize; i++)
        {
            fprintf(file, "%f\n", (HI_FLOAT)result1[i] / SVP_WK_QUANT_BASE);
        }
        fclose(file);
        file = NULL;

        /******************* write output_box **********************/
        file_path = resultPath + imgNamePair.first + "_output_box.txt";
        file = fopen(file_path.c_str(), "w");
        SVP_CHECK(NULL != file, HI_FAILURE);

        for (HI_U32 i = 0; i < model_info.u32MaxRoiFrameCnt * model_info.u32ClassSize * SVP_WK_COORDI_NUM; i++)
        {
            fprintf(file, "%d\n", result2[i]);
        }
        fclose(file);
        file = NULL;

        /******************* write output_count **********************/
        file_path = resultPath + imgNamePair.first + "_output_count.txt";
        file = fopen(file_path.c_str(), "w");
        SVP_CHECK(NULL != file, HI_FAILURE);

        for (HI_U32 i = 0; i < model_info.u32ClassSize; i++)
        {
            fprintf(file, "%d\n", result3[i]);
        }

        fclose(file);
        file = NULL;

#endif

    return HI_SUCCESS;
}

HI_U32 GetRFCNAssistMemSize(stRFCNPara* para)
{
    HI_U32 u32NumAnchors = (para->model_info.u32NumRatioAnchors) *
                           (para->model_info.u32NumScaleAnchors)*
                           (para->model_info.astReportNodeInfo[0].u32ConvHeight) *
                           (para->model_info.astReportNodeInfo[0].u32ConvWidth);

    HI_U32 u32AnchorSize    = u32NumAnchors* SVP_WK_COORDI_NUM * sizeof(HI_U32);
    HI_U32 u32BboxDeltaSize = u32AnchorSize;
    HI_U32 u32ProposalSize  = u32NumAnchors * SVP_WK_PROPOSAL_WIDTH * sizeof(HI_U32);
    HI_U32 u32RatioSize     = (para->model_info.u32NumRatioAnchors) * SVP_WK_COORDI_NUM * sizeof(HI_FLOAT);
    HI_U32 u32ScaleSize     = (para->model_info.u32NumRatioAnchors) *
                              (para->model_info.u32NumScaleAnchors) * SVP_WK_COORDI_NUM * sizeof(HI_FLOAT);
    HI_U32 u32ScoresSize    = u32NumAnchors * SVP_WK_SCORE_NUM * sizeof(HI_FLOAT);
    HI_U32 u32StackSize     = MAX_STACK_DEPTH * sizeof(NNIE_STACK_S);

    HI_U32 u32TotalSize = u32AnchorSize + u32BboxDeltaSize + u32ProposalSize + u32RatioSize +
                          u32ScaleSize + u32ScoresSize + u32StackSize;

    return u32TotalSize;
}
