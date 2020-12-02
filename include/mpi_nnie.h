/******************************************************************************

  Copyright (C), 2017-2018, Hisilicon Tech. Co., Ltd.

 ******************************************************************************
  File Name     : mpi_nnie.h
  Version       : Initial Draft
  Author        : Hisilicon multimedia software (SVP) group
  Created       : 2017/06/06
  Description   :
  History       :
  1.Date        : 2017/06/06
    Author      :
    Modification: Created file

******************************************************************************/
#ifndef _HI_MPI_NNIE_H_
#define _HI_MPI_NNIE_H_

#ifdef __cplusplus
#if __cplusplus
extern "C"{
#endif
#endif /* End of #ifdef __cplusplus */

#include "hi_nnie.h"

////加载模型到nnie框架，参数：内存信息，nnie框架模型
HI_S32 HI_MPI_SVP_NNIE_LoadModel(const SVP_SRC_MEM_INFO_S *pstModelBuf, SVP_NNIE_MODEL_S *pstModel);


//申请任务存储
HI_S32 HI_MPI_SVP_NNIE_GetTskBufSize(HI_U32 u32MaxInputNum, HI_U32 u32MaxBboxNum,
       const SVP_NNIE_MODEL_S *pstModel, HI_U32 au32TskBufSize[], HI_U32 u32NetSegNum);

/*****************************************************************************
*   Prototype    : HI_MPI_SVP_NNIE_Forward
*   Description  : Perform CNN prediction on input sample(s), and output responses for corresponding sample(s)
*   Parameters   : SVP_NNIE_HANDLE                  *phSvpNnieHandle      Returned handle ID of a task
*                  const SVP_SRC_BLOB_S                    astSrc[]             Input node array.
*                  const SVP_NNIE_MODEL_S                  pstModel             CNN model data
*                  const SVP_DST_BLOB_S                    astDst[]             Output node array
*                  const SVP_NNIE_FORWARD_CTRL_S          *pstForwardCtrl       Ctrl prameters
*                  HI_BOOL                           bInstant             Flag indicating whether to generate an interrupt.
*                                                                         If the output result blocks the next operation,
*                                                                         set bInstant to HI_TRUE.
*   Spec         :
*   Return Value: HI_SUCCESS: Success; Error codes: Failure.
*   Spec         :
*   History:
*
*       1.  Date         : 2017-06-06
*          Author        :
*          Modification  : Created function
*   任务处理ID，输入节点，模型数据，输出节点，参数，常量
*****************************************************************************/
HI_S32 HI_MPI_SVP_NNIE_Forward(SVP_NNIE_HANDLE *phSvpNnieHandle,
    const SVP_SRC_BLOB_S astSrc[], const SVP_NNIE_MODEL_S *pstModel,const SVP_DST_BLOB_S astDst[],
    const SVP_NNIE_FORWARD_CTRL_S *pstForwardCtrl,HI_BOOL bInstant);

/*****************************************************************************
*   Prototype    : HI_MPI_SVP_NNIE_ForwardWithBbox
*   Description  : Perform CNN prediction on input sample(s), and output responses for corresponding sample(s)
*   Parameters   : SVP_NNIE_HANDLE                  *pNnieHandle        Returned handle ID of a task
*                  const SVP_SRC_BLOB_S                    astSrc[]           Input nodes' array.
*                  const SVP_SRC_BLOB_S                    astBbox[]          Input nodes' Bbox array.
*                  const SVP_NNIE_MODEL_S                  pstModel           CNN model data
*                  const SVP_DST_BLOB_S                    astDst[]           Output node array
*                  const SVP_NNIE_FORWARD_WITHBBOX_CTRL_S *pstForwardCtrl     Ctrl prameters
*                  HI_BOOL                           bInstant           Flag indicating whether to generate an interrupt.
*                                                                       If the output result blocks the next operation,
*                                                                       set bInstant to HI_TRUE.
*   Spec         :
*   Return Value: HI_SUCCESS: Success; Error codes: Failure.
*   Spec         :
*   History:
*
*       1.  Date         : 2017-08-09
*          Author        :
*          Modification  : Created function
*   任务处理ID，输入节点，输入bbox，模型数据，输出节点，参数，常量
*****************************************************************************/
HI_S32 HI_MPI_SVP_NNIE_ForwardWithBbox(SVP_NNIE_HANDLE *phSvpNnieHandle,
    const SVP_SRC_BLOB_S astSrc[], const SVP_SRC_BLOB_S astBbox[], const SVP_NNIE_MODEL_S *pstModel,
    const SVP_DST_BLOB_S astDst[], const SVP_NNIE_FORWARD_WITHBBOX_CTRL_S *pstForwardCtrl,HI_BOOL bInstant);

/*****************************************************************************
*   Prototype    : HI_MPI_SVP_NNIE_UnloadModel
*   Description  : Unload cnn model
*   Parameters   : SVP_NNIE_MODEL_S         *pstModel          Output model
*
*   Return Value : HI_SUCCESS: Success; Error codes: Failure.
*   Spec         :
*   History:
*
*       1.  Date         : 2017-06-06
*           Author       :
*           Modification : Created function
*   卸载模型
*****************************************************************************/
 HI_S32 HI_MPI_SVP_NNIE_UnloadModel(SVP_NNIE_MODEL_S *pstModel);

/*****************************************************************************
*   Prototype    : HI_MPI_SVP_NNIE_Query
*   Description  : This API is used to query the status of a function runed on nnie.
                   In block mode, the system waits until the function that is being queried is called.
                   In non-block mode, the current status is queried and no action is taken.
                   查询函数状态
*   Parameters   : SVP_NNIE_ID_E        enNnieId       NNIE Id
*                  SVP_NNIE_HANDLE      nnieHandle     nnieHandle of a called function. It is entered by users.
*                  HI_BOOL                    *pbFinish      Returned status
*                  HI_BOOL              bBlock         Flag indicating the block mode or non-block mode
*   Return Value : HI_SUCCESS: Success;Error codes: Failure.
*   Spec         :
*   History:
*
*       1.  Date         : 2017-06-06
*          Author        :
*          Modification  : Created function
*   nnit的ID，手柄，返回状态，返回模式（block or non-block）
*****************************************************************************/
HI_S32 HI_MPI_SVP_NNIE_Query(SVP_NNIE_ID_E enNnieId,SVP_NNIE_HANDLE svpNnieHandle,HI_BOOL *pbFinish,HI_BOOL bBlock);



#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif
#endif/*_HI_MPI_NNIE_H_*/

