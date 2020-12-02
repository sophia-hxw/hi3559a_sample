#ifndef __HI_SVP_SAMPLE_COM_H__
#define __HI_SVP_SAMPLE_COM_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <string>
#include <utility>

#include "hi_type.h"
#include "hi_nnie.h"

#include "SvpSampleWk.h"

using namespace std;

#ifdef _WIN32
#include <direct.h>
#define SVP_SAMPLE_MAX_PATH  _MAX_PATH
#else
#include <sys/stat.h>
#define _mkdir(a) mkdir((a),0755)

#include <linux/limits.h>
#define SVP_SAMPLE_MAX_PATH  PATH_MAX
#endif

#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#endif


#ifdef _MSC_VER
#include <Windows.h>

// the windows Sleep function used millisecond, and it's accuracy time is about 10ms.
// you should re-defined the micro when you want more accuracy delay like linux usleep.
#define USLEEP(microsecend) Sleep((microsecend/1000)?(microsecend/1000):(1))

#define FSCANF_S(stream, fmt, ...)  fscanf_s(stream, fmt, ##__VA_ARGS__)

#define TRACE(...) fprintf(stderr, __VA_ARGS__)

#define CHECK_EXP_RET(exp, ret, ...)\
{\
    if(exp)\
    {\
        TRACE("[File]:%s, [Line]:%d, [Error]: ", __FILE__, __LINE__);\
        TRACE(__VA_ARGS__);\
        TRACE("\n");\
        return ret;\
    }\
}

#define CHECK_EXP_GOTO(exp, flag, ...)\
{\
    if (exp)\
    {\
        TRACE("[File]:%s, [Line]:%d, [Error] ", __FILE__, __LINE__);\
        TRACE(__VA_ARGS__);\
        TRACE("\n");\
        goto flag;\
    }\
}

#else

#include <unistd.h>

// the windows Sleep function used millisecond, and it's accuracy time is about 10ms.
#define USLEEP(microsecend) usleep(microsecend)

#define FSCANF_S(stream, fmt, ...)  fscanf(stream, fmt, ##__VA_ARGS__)

#define TRACE(fmt...) fprintf(stderr, fmt)

#define CHECK_EXP_RET(exp, ret, fmt...)\
{\
    if(exp)\
    {\
        TRACE("[File]:%s, [Line]:%d, [Error]: ", __FILE__, __LINE__);\
        TRACE(fmt);\
        TRACE("\n");\
        return ret;\
    }\
}

#define CHECK_EXP_GOTO(exp, flag, fmt...)\
{\
    if (exp)\
    {\
        TRACE("[File]:%s, [Line]:%d, [Error] ", __FILE__, __LINE__);\
        TRACE(fmt);\
        TRACE("\n");\
        goto flag;\
    }\
}
#endif


//对齐到32位，干嘛的？不知道？
HI_U32 SvpSampleAlign(HI_U32 u32Size, HI_U32 u32AlignNum);

//分配u32Size大小的内存，依赖环境选择32位或者64位
HI_S32 SvpSampleMalloc(HI_CHAR *pchMmb, HI_CHAR *pchZone,HI_U64 *pu64PhyAddr, HI_VOID **ppvVirAddr, HI_U32 u32Size);
HI_S32 SvpSampleMallocMem(HI_CHAR *pchMmb, HI_CHAR *pchZone, HI_U32 u32Size, SVP_MEM_INFO_S *pstMem);

//Malloc mem with cache,depend on different environment
HI_S32 SvpSampleMalloc_Cached(HI_CHAR *pchMmb, HI_CHAR *pchZone, HI_U64 *pu64PhyAddr, HI_VOID **ppvVirAddr, HI_U32 u32Size);
HI_S32 SvpSampleMallocMemCached(HI_CHAR *pchMmb, HI_CHAR *pchZone, HI_U32 u32Size, SVP_MEM_INFO_S *pstMem);

//Flush cache, if u32PhyAddr==0£¬that means flush all cache
//清理缓存？
HI_S32 SvpSampleFlushCache(HI_U64 u64PhyAddr, HI_VOID *pvVirAddr, HI_U32 u32Size);
HI_S32 SvpSampleFlushMemCache(SVP_MEM_INFO_S *pstMem);

//Free mem,depend on different environment
//释放不同环境申请的内存
HI_VOID SvpSampleFree(HI_U64 u64PhyAddr, HI_VOID *pvVirAddr);
HI_VOID SvpSampleMemFree(SVP_MEM_INFO_S *pstMem);

//打开文件，环境相关
FILE* SvpSampleOpenFile(const HI_CHAR *pchFileName, const HI_CHAR *pchMode);

//关闭文件
HI_VOID SvpSampleCloseFile(FILE *fp);

//分配块ncwh的内存，按行存储
HI_S32 SvpSampleMallocBlob(SVP_BLOB_S *pstBlob, SVP_BLOB_TYPE_E enType, HI_U32 u32Num, HI_U32 u32Chn,
    HI_U32 u32Width, HI_U32 u32Height, HI_U32 u32UsrAlign = STRIDE_ALIGN);

//lstm中用到的，需加上lstm的ctx参数文件内参数
HI_S32 SvpSampleMallocSeqBlob(SVP_BLOB_S *pstBlob, SVP_BLOB_TYPE_E enType, HI_U32 u32Num, HI_U32 u32Dim,
    SVP_SAMPLE_LSTMRunTimeCtx *pstLSTMCtx);

//释放块内存
void SvpSampleFreeBlob(SVP_BLOB_S *pstBlob);

//分配rpn块内存
HI_S32 SvpSampleMallocRPNBlob(SVP_BLOB_S *pstBlob, HI_U32 u32Size, HI_U32 u32UsrStride = STRIDE_ALIGN);

//释放rpn块内存
void SvpSampleFreeRPNBlob(SVP_BLOB_S *pstBlob);

/*SVP_SAMPLE_FILE_NAME_PAIR first:  basic filename, second: filename suffix*/
//文件名称对，名字+后缀
typedef pair<string, string> SVP_SAMPLE_FILE_NAME_PAIR;

//从一个文件读一张图，格式(U8/YVU420SP/YVU422SP/S32/VEC_S32/SEQ_S32)
HI_S32 SvpSampleImgReadFromImglist(FILE *fp, SVP_BLOB_S *pstBlob, HI_U32 u32StartLine,
    vector<SVP_SAMPLE_FILE_NAME_PAIR>& imgNameRecoder);

//从多个文件读u32Num张图，每个image/featuremaps/vectors都是二进制文件，
//imagelist的一行就代表一个二进制文件
HI_S32 SvpSampleReadAllSrcImg(FILE *afp[], SVP_SRC_BLOB_S astSrcBlobs[], HI_U32 u32SrcNum,
    vector<SVP_SAMPLE_FILE_NAME_PAIR>& imgNameRecoder);

//从图片path中获取文件名称对
SVP_SAMPLE_FILE_NAME_PAIR SvpSampleGetFileNameFromPath(std::string& strImgPath);


#endif //__HI_SVP_SAMPLE_COM_H__