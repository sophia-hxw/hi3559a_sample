/*
*** 注释标签说明：
*** QUES: 没看懂的问题，待解释？
*** TODO: 懂抽象层面含义，需要补充细节
*/

#ifndef __HI_COMM_SVP_H__
#define __HI_COMM_SVP_H__

#ifdef __cplusplus
#if __cplusplus
extern "C"{
#endif
#endif /* __cplusplus */

#include "hi_type.h"
#include "hi_errno.h"

//blob类型
typedef enum hiSVP_BLOB_TYPE_E
{
    SVP_BLOB_TYPE_S32       =  0x0,

    SVP_BLOB_TYPE_U8        =  0x1,

    /*channel = 3*/
    SVP_BLOB_TYPE_YVU420SP  =  0x2,
    /*channel = 3*/
    SVP_BLOB_TYPE_YVU422SP  =  0x3,

    SVP_BLOB_TYPE_VEC_S32   =  0x4,

    SVP_BLOB_TYPE_SEQ_S32   =  0x5,

    SVP_BLOB_TYPE_BUTT
}SVP_BLOB_TYPE_E;

//caffe中的blob，也就是[N，C，H，W]
typedef struct hiSVP_BLOB_S
{
     SVP_BLOB_TYPE_E enType;     //类型，枚举结构体
     HI_U32 u32Stride;           /*Stride, a line bytes num*///QUES: strade=sizeof(element)*weight?

     HI_U64 u64VirAddr;          //虚拟内存地址
     HI_U64 u64PhyAddr;          //物理内存地址

    HI_U32      u32Num;         //caffe中的N，对应帧数或者sequence数量
    union
    {
        struct
        {
            HI_U32 u32Width;    /*W: frame width, correspond to caffe blob's w*/
            HI_U32 u32Height;   /*H: frame height, correspond to caffe blob's h*/
            HI_U32 u32Chn;      /*C: frame channel,correspond to caffe blob's c*/
        }stWhc;
        struct
        {
            HI_U32 u32Dim;          /*D: vector dimension*/
            HI_U64 u64VirAddrStep;  /*T: virtual address of time steps array in each sequence*/
        }stSeq;
    }unShape;
}SVP_BLOB_S;

typedef SVP_BLOB_S  SVP_SRC_BLOB_S;
typedef SVP_BLOB_S  SVP_DST_BLOB_S;

//存储信息结构体
typedef struct hiSVP_MEM_INFO_S
{
    HI_U64  u64PhyAddr; //物理存储地址
    HI_U64  u64VirAddr; //虚拟存储地址
    HI_U32  u32Size;    //大小
}SVP_MEM_INFO_S;

typedef SVP_MEM_INFO_S SVP_SRC_MEM_INFO_S;
typedef SVP_MEM_INFO_S SVP_DST_MEM_INFO_S;

/*Image type*/
typedef enum hiSVP_IMAGE_TYPE_E
{
   SVP_IMAGE_TYPE_U8C1           =  0x0,
   SVP_IMAGE_TYPE_S8C1           =  0x1,

   SVP_IMAGE_TYPE_YUV420SP       =  0x2,       /*YUV420 SemiPlanar*/
   SVP_IMAGE_TYPE_YUV422SP       =  0x3,       /*YUV422 SemiPlanar*/
   SVP_IMAGE_TYPE_YUV420P        =  0x4,       /*YUV420 Planar */
   SVP_IMAGE_TYPE_YUV422P        =  0x5,       /*YUV422 planar */

   SVP_IMAGE_TYPE_S8C2_PACKAGE   =  0x6,
   SVP_IMAGE_TYPE_S8C2_PLANAR    =  0x7,

   SVP_IMAGE_TYPE_S16C1          =  0x8,
   SVP_IMAGE_TYPE_U16C1          =  0x9,

   SVP_IMAGE_TYPE_U8C3_PACKAGE   =  0xa,
   SVP_IMAGE_TYPE_U8C3_PLANAR    =  0xb,

   SVP_IMAGE_TYPE_S32C1          =  0xc,
   SVP_IMAGE_TYPE_U32C1          =  0xd,

   SVP_IMAGE_TYPE_S64C1          =  0xe,
   SVP_IMAGE_TYPE_U64C1          =  0xf,

   SVP_IMAGE_TYPE_BUTT

}SVP_IMAGE_TYPE_E;

/*Image*/
typedef struct hiSVP_IMAGE_S
{
    HI_U64  au64PhyAddr[3]; /* RW;The physical address of the image */
    HI_U64  au64VirAddr[3]; /* RW;The virtual address of the image */
    HI_U32  au32Stride[3];  /* RW;The stride of the image */
    HI_U32  u32Width;       /* RW;The width of the image */
    HI_U32  u32Height;      /* RW;The height of the image */
    SVP_IMAGE_TYPE_E  enType; /* RW;The type of the image */
}SVP_IMAGE_S;

typedef SVP_IMAGE_S SVP_SRC_IMAGE_S;
typedef SVP_IMAGE_S SVP_DST_IMAGE_S;

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* __cplusplus */


#endif /* __HI_COMM_SVP_H__ */
