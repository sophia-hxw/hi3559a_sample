#ifndef __SVP_SAMPLE_YOLO_H__
#define __SVP_SAMPLE_YOLO_H__

#include <string>
#include <vector>

#include "hi_type.h"
#include "hi_nnie.h"
#include "detectionCom.h"
#include "SvpSampleWk.h"

/* YOLO V1 */
#define SVP_SAMPLE_YOLOV1_IMG_WIDTH                (448)
#define SVP_SAMPLE_YOLOV1_IMG_HEIGHT               (448)

#define SVP_SAMPLE_YOLOV1_BBOX_CNT                 (98)
#define SVP_SAMPLE_YOLOV1_CLASS_CNT                (20)
#define SVP_SAMPLE_YOLOV1_THRESHOLD                (0.2)
#define SVP_SAMPLE_YOLOV1_IOU                      (0.5)

#define SVP_SAMPLE_YOLOV1_GRID_NUM                 (7)
#define SVP_SAMPLE_YOLOV1_GRID_SQR_NUM             (SVP_SAMPLE_YOLOV1_GRID_NUM * SVP_SAMPLE_YOLOV1_GRID_NUM)
#define SVP_SAMPLE_YOLOV1_CHANNEL_NUM              (30)
#define SVP_SAMPLE_YOLOV1_CHANNEL_GRID_NUM         (SVP_SAMPLE_YOLOV1_CHANNEL_NUM * SVP_SAMPLE_YOLOV1_GRID_SQR_NUM)
#define SVP_SAMPLE_YOLOV1_BOX_NUM                  (2)

void SvpSampleWkYoloV1GetResult(SVP_BLOB_S *pstDstBlob, SVP_SAMPLE_BOX_RESULT_INFO_S *pstBoxes,
    HI_U32 *pu32BoxNum, std::string& strResultFolderDir, std::vector<SVP_SAMPLE_FILE_NAME_PAIR>& imgNameRecoder);

/* YOLO V2 */
#define SVP_SAMPLE_YOLOV2_IMG_WIDTH                (416)
#define SVP_SAMPLE_YOLOV2_IMG_HEIGHT               (416)

#define SVP_SAMPLE_YOLOV2_GRIDNUM                  (13)
#define SVP_SAMPLE_YOLOV2_GRIDNUM_SQR              (SVP_SAMPLE_YOLOV2_GRIDNUM * SVP_SAMPLE_YOLOV2_GRIDNUM)
#define SVP_SAMPLE_YOLOV2_CHANNLENUM               (50)
#define SVP_SAMPLE_YOLOV2_PARAMNUM                 (10)
#define SVP_SAMPLE_YOLOV2_BOXNUM                   (5)
#define SVP_SAMPLE_YOLOV2_CLASSNUM                 (5)
#define SVP_SAMPLE_YOLOV2_BOXTOTLENUM              (SVP_SAMPLE_YOLOV2_GRIDNUM * SVP_SAMPLE_YOLOV2_GRIDNUM * SVP_SAMPLE_YOLOV2_BOXNUM)
#define SVP_SAMPLE_YOLOV2_MAX_BOX_NUM              (10)
#define SVP_SAMPLE_YOLOV2_WIDTH                    (7)

void SvpSampleWkYoloV2GetResult(SVP_BLOB_S *pstDstBlob, HI_S32 *ps32ResultMem,
    SVP_SAMPLE_BOX_RESULT_INFO_S *pstBoxes, HI_U32 *pu32BoxNum, std::string& strResultFolderDir,
    std::vector<SVP_SAMPLE_FILE_NAME_PAIR>& imgNameRecoder);

/* YOLO V3 */
// #define SVP_SAMPLE_YOLOV3_SRC_WIDTH                (416)
// #define SVP_SAMPLE_YOLOV3_SRC_HEIGHT               (416)

// #define SVP_SAMPLE_YOLOV3_GRIDNUM_CONV_82          (13)
// #define SVP_SAMPLE_YOLOV3_GRIDNUM_CONV_94          (26)
// #define SVP_SAMPLE_YOLOV3_GRIDNUM_CONV_106         (52)
// #define SVP_SAMPLE_YOLOV3_RESULT_BLOB_NUM          (3) //3个先验框
// #define SVP_SAMPLE_YOLOV3_CHANNLENUM               (255)
// #define SVP_SAMPLE_YOLOV3_PARAMNUM                 (85)//？
// #define SVP_SAMPLE_YOLOV3_BOXNUM                   (3)
// #define SVP_SAMPLE_YOLOV3_CLASSNUM                 (80)
// #define SVP_SAMPLE_YOLOV3_MAX_BOX_NUM              (10)

/* YOLO V4 */
#define SVP_SAMPLE_YOLOV3_SRC_WIDTH                (608)
#define SVP_SAMPLE_YOLOV3_SRC_HEIGHT               (608)

#define SVP_SAMPLE_YOLOV3_GRIDNUM_CONV_82          (76)
#define SVP_SAMPLE_YOLOV3_GRIDNUM_CONV_94          (38)
#define SVP_SAMPLE_YOLOV3_RESULT_BLOB_NUM          (2) //3个先验框
#define SVP_SAMPLE_YOLOV3_CHANNLENUM               (60)
#define SVP_SAMPLE_YOLOV3_PARAMNUM                 (15)//？
#define SVP_SAMPLE_YOLOV3_BOXNUM                   (4)
#define SVP_SAMPLE_YOLOV3_CLASSNUM                 (10)
#define SVP_SAMPLE_YOLOV3_MAX_BOX_NUM              (10)

typedef enum hiSVP_SAMPLE_YOLOV3_SCALE_TYPE
{
    CONV_82 = 0,
    CONV_94,
   // CONV_106,
    SVP_SAMPLE_YOLOV3_SCALE_TYPE_MAX
}SVP_SAMPLE_YOLOV3_SCALE_TYPE_E;

void SvpSampleWkYoloV3GetResult(SVP_BLOB_S *pstDstBlob, HI_S32 *ps32ResultMem,
    SVP_SAMPLE_BOX_RESULT_INFO_S *pstBoxes, HI_U32 *pu32BoxNum, std::string& strResultFolderDir,
    std::vector<SVP_SAMPLE_FILE_NAME_PAIR>& imgNameRecoder);

HI_U32 SvpSampleGetYolov3ResultMemSize(SVP_SAMPLE_YOLOV3_SCALE_TYPE_E enScaleType);

#endif //__SVP_SAMPLE_WK_H__
