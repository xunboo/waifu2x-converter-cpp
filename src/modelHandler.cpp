/*
 * modelHandler.cpp
 *
 *  Created on: 2015/05/24
 *      Author: wlamigo
 */

#include "modelHandler.hpp"
// #include <iostream> in modelHandler.hpp
#include "cvwrap.hpp"
#include <fstream>
#include <thread>
#include <atomic>
#include "sec.hpp"
//#include "threadPool.hpp"
#include "common.hpp"
#include "filters.hpp"
#include "params.h"

namespace w2xc {

int Model::getNInputPlanes() {
	return nInputPlanes;
}

int Model::getNOutputPlanes() {
	return nOutputPlanes;
}

int Model::getStrideSize() {
	return strideSize;
}

int Model::getKernelSize() {
	return kernelSize;
}

int Model::getPadSize() {
	return padSize;
}


bool
Model::filter_CPU(ComputeEnv *env,
		 Buffer *packed_input_buf,
		 Buffer *packed_output_buf,
		 const W2Size &size)
{
	size_t in_size = sizeof(float) * size.width * size.height * nInputPlanes;
	const float *packed_input = (float*)packed_input_buf->get_read_ptr_host(env, in_size);
	float *packed_output = (float*)packed_output_buf->get_write_ptr_host(env);
	
	std::atomic<int> yi_shared(0);

	auto thread_func = [&](){
		int w = size.width;
		int h = size.height;

		while (1) {
			int yi = yi_shared++;

			if (yi >= h) {
				break;
			}

			float *out_line = packed_output + w*nOutputPlanes * yi;

			int yi0 = yi-1;
			int yi1 = yi;
			int yi2 = yi+1;
			int yi3 = yi+2;

			if (yi == 0) {
				yi0 = 0;
			}
			if (yi == h-2) {
				yi3 = yi1;
			} 
			else if (yi == h-1) {
				yi2 = yi1;
				yi3 = yi1;
			}

			const float *in_line0 = packed_input + w * nInputPlanes * yi0;
			const float *in_line1 = packed_input + w * nInputPlanes * yi1;
			const float *in_line2 = packed_input + w * nInputPlanes * yi2;
			const float *in_line3 = packed_input + w * nInputPlanes * yi3;

			for (int xi=0; xi<w; xi++) {
				int x0 = xi-1;
				int x1 = xi;
				int x2 = xi+1;
				int x3 = xi+2;

				if (xi == 0) {
					x0 = 0;
				}

				if (xi == w-2) {
					x3 = x1;
				}
				else if (xi == w-1) {
					x2 = x1;
					x3 = x1;
				}

				const float *in00 = in_line0 + x0 * nInputPlanes;
				const float *in01 = in_line0 + x1 * nInputPlanes;
				const float *in02 = in_line0 + x2 * nInputPlanes;
				const float *in03 = in_line0 + x3 * nInputPlanes;

				const float *in10 = in_line1 + x0 * nInputPlanes;
				const float *in11 = in_line1 + x1 * nInputPlanes;
				const float *in12 = in_line1 + x2 * nInputPlanes;
				const float *in13 = in_line1 + x3 * nInputPlanes;

				const float *in20 = in_line2 + x0 * nInputPlanes;
				const float *in21 = in_line2 + x1 * nInputPlanes;
				const float *in22 = in_line2 + x2 * nInputPlanes;
				const float *in23 = in_line2 + x3 * nInputPlanes;

				const float *in30 = in_line3 + x0 * nInputPlanes;
				const float *in31 = in_line3 + x1 * nInputPlanes;
				const float *in32 = in_line3 + x2 * nInputPlanes;
				const float *in33 = in_line3 + x3 * nInputPlanes;

				for (int oi=0; oi<nOutputPlanes; oi++) {
					float sum = 0;

					for (int ii=0; ii<nInputPlanes; ii++) {
						int wMatIndex = nInputPlanes * oi + ii;
						const float *w = weights[wMatIndex].ptr<float>(0);
						
						if(kernelSize == 3){
							sum += in00[ii] * w[0];
							sum += in01[ii] * w[1];
							sum += in02[ii] * w[2];

							sum += in10[ii] * w[3];
							sum += in11[ii] * w[4];
							sum += in12[ii] * w[5];

							sum += in20[ii] * w[6];
							sum += in21[ii] * w[7];
							sum += in22[ii] * w[8];
						}
						else if(kernelSize == 4){
							sum += in00[ii] * w[0];
							sum += in01[ii] * w[1];
							sum += in02[ii] * w[2];
							sum += in03[ii] * w[3];

							sum += in10[ii] * w[4];
							sum += in11[ii] * w[5];
							sum += in12[ii] * w[6];
							sum += in13[ii] * w[7];

							sum += in20[ii] * w[8];
							sum += in21[ii] * w[9];
							sum += in22[ii] * w[10];
							sum += in23[ii] * w[11];

							sum += in30[ii] * w[12];
							sum += in31[ii] * w[13];
							sum += in32[ii] * w[14];
							sum += in33[ii] * w[15];
						}
					}

					float v = sum;
					v += (float) biases[oi];
					float mtz = (std::max)(v, 0.0f);
					float ltz = (std::min)(v, 0.0f);
					v = ltz*0.1f + mtz;

					out_line[xi*nOutputPlanes + oi] = v;
				}
			}
		}
	};

	std::vector<std::thread> workerThreads;
	int nJob = modelUtility::getInstance().getNumberOfJobs();
	for (int ji=0; ji<nJob; ji++) {
		workerThreads.emplace_back(std::thread(thread_func));
	}

	for (auto&th : workerThreads) {
		th.join();
	}
	return true;
}

//#define COMPARE_RESULT

bool Model::filter_AVX_OpenCL(W2XConv *conv,
			      ComputeEnv *env,
			      Buffer *packed_input_buf,
			      Buffer *packed_output_buf,
			      const W2Size &size)
{
	int vec_width;
	int weight_step;
	int nJob = modelUtility::getInstance().getNumberOfJobs();
	const struct W2XConvProcessor *proc = conv->target_processor;

	bool gpu = (proc->type == W2XCONV_PROC_OPENCL) || (proc->type == W2XCONV_PROC_CUDA);

	if (gpu) {
		weight_step = GPU_VEC_WIDTH;
		vec_width = GPU_VEC_WIDTH;
	} else {
		weight_step = nOutputPlanes;
		vec_width = VEC_WIDTH;
	}

	float *weight_flat = (float*)w2xc_aligned_malloc(sizeof(float)*nInputPlanes*weight_step*kernelSize*kernelSize, 64);
	float *fbiases_flat = (float*)w2xc_aligned_malloc(sizeof(float) * biases.size(), 64);

	for (int i=0; i<(int)biases.size(); i++) {
		fbiases_flat[i] = (float) biases[i];
	}

	if (nOutputPlanes == 1) {
		if (gpu) {
			for (int ii=0; ii<nInputPlanes; ii++) {
				W2Mat &wm = weights[ii];
				const float *src0 = wm.ptr<float>(0);
				const float *src1 = wm.ptr<float>(1);
				const float *src2 = wm.ptr<float>(2);

				float *dst = weight_flat + ii * 9;
				dst[0] = src0[0];
				dst[1] = src0[1];
				dst[2] = src0[2];

				dst[3] = src1[0];
				dst[4] = src1[1];
				dst[5] = src1[2];

				dst[6] = src2[0];
				dst[7] = src2[1];
				dst[8] = src2[2];

			}
		} else {
			for (int ii=0; ii<nInputPlanes; ii++) {
				W2Mat &wm = weights[ii];
				const float *src0 = wm.ptr<float>(0);
				const float *src1 = wm.ptr<float>(1);
				const float *src2 = wm.ptr<float>(2);

				int ii_0 = ii % vec_width;
				int ii_1 = (ii / vec_width) * vec_width;

				float *dst = weight_flat + ii_1 * 9  + ii_0;
				dst[0 * vec_width] = src0[0];
				dst[1 * vec_width] = src0[1];
				dst[2 * vec_width] = src0[2];

				dst[3 * vec_width] = src1[0];
				dst[4 * vec_width] = src1[1];
				dst[5 * vec_width] = src1[2];

				dst[6 * vec_width] = src2[0];
				dst[7 * vec_width] = src2[1];
				dst[8 * vec_width] = src2[2];
			}
		}
	} else if (gpu && nInputPlanes == 1) {
		for (int oi=0; oi<nOutputPlanes; oi++) {
			W2Mat &wm = weights[oi];
			const float *src0 = wm.ptr<float>(0);
			const float *src1 = wm.ptr<float>(1);
			const float *src2 = wm.ptr<float>(2);

			float *dst = weight_flat + oi * 9;
			dst[0] = src0[0];
			dst[1] = src0[1];
			dst[2] = src0[2];

			dst[3] = src1[0];
			dst[4] = src1[1];
			dst[5] = src1[2];

			dst[6] = src2[0];
			dst[7] = src2[1];
			dst[8] = src2[2];
		}
	} else if (nOutputPlanes == 3) {
		/* |       o0        |       o1        | o2 ... |
		 * |i0 i1 i2 ... i127|i0 i1 i2 ... i127| ...    |*/

		for (int oi=0; oi<nOutputPlanes; oi++) {
			for (int ii=0; ii<nInputPlanes; ii++) {
				int mi = oi*nInputPlanes+ii;
				W2Mat &wm = weights[mi];
				const float *src0 = wm.ptr<float>(0);
				const float *src1 = wm.ptr<float>(1);
				const float *src2 = wm.ptr<float>(2);

				float *dst = weight_flat + (oi * nInputPlanes * 9) + ii;
				dst[0*nInputPlanes] = src0[0];
				dst[1*nInputPlanes] = src0[1];
				dst[2*nInputPlanes] = src0[2];

				dst[3*nInputPlanes] = src1[0];
				dst[4*nInputPlanes] = src1[1];
				dst[5*nInputPlanes] = src1[2];

				dst[6*nInputPlanes] = src2[0];
				dst[7*nInputPlanes] = src2[1];
				dst[8*nInputPlanes] = src2[2];
			}
		}
	} else if (gpu && (nInputPlanes == 3) && (nOutputPlanes == 32)) {
		/* | i0             | i1        | i2 .. iN-1|
		 * |o0 o1 o2 o3..o31|o0 .... o32| ....      |
		 * |<-            ->|
		 * |    32          |
		 * |   x  9         |
		 */

		for (int oi=0; oi<nOutputPlanes; oi++) {
			for (int ii=0; ii<nInputPlanes; ii++) {
				int mi = oi*nInputPlanes+ii;
				W2Mat &wm = weights[mi];
				const float *src0 = wm.ptr<float>(0);
				const float *src1 = wm.ptr<float>(1);
				const float *src2 = wm.ptr<float>(2);

				float *dst = weight_flat + (ii * nOutputPlanes * 9) + oi;
				dst[0*nOutputPlanes] = src0[0];
				dst[1*nOutputPlanes] = src0[1];
				dst[2*nOutputPlanes] = src0[2];

				dst[3*nOutputPlanes] = src1[0];
				dst[4*nOutputPlanes] = src1[1];
				dst[5*nOutputPlanes] = src1[2];

				dst[6*nOutputPlanes] = src2[0];
				dst[7*nOutputPlanes] = src2[1];
				dst[8*nOutputPlanes] = src2[2];
			}
		}
	} else {
		bool simd_oplane = false;
		bool simd_iplane = false;
		int simd_vec_width = 0;
		if (proc->type == W2XCONV_PROC_HOST) {
			switch (proc->sub_type) {
			case W2XCONV_PROC_HOST_SSE3:
				simd_vec_width = 4;
				simd_oplane = true;
				break;

			case W2XCONV_PROC_HOST_NEON:
				simd_vec_width = 4;
				simd_oplane = true;
				break;


			case W2XCONV_PROC_HOST_AVX:
			case W2XCONV_PROC_HOST_FMA:
				simd_vec_width = 8;
				simd_oplane = true;
				break;
			}
		}

		simd_oplane = simd_oplane && (nInputPlanes%(simd_vec_width*4) == 0) && (nOutputPlanes%(simd_vec_width*2) == 0);
		simd_iplane = simd_iplane && (nInputPlanes%(simd_vec_width*4) == 0) && (nOutputPlanes%(simd_vec_width*2) == 0);

		if (simd_oplane || simd_iplane) {
			/* 
			 * weight_chunk (16x32x3x4 = 6144[Byte])
			 * (where op_block_size=16, ip_block_size=32)
			 *
			 * 111                                            oplane x16
			 * 16 16 .. (x16)  ..16                           iplane x32
			 *            \               |               /   horiz  x3
			 *                                                oplane xnOutputPlane_block
			 *                                                iplane xnInputPlane_block
			 *                                                vert   x3
			 */
			int ip_block_size;
			int op_block_size;

			if (simd_oplane) {
				ip_block_size = (simd_vec_width*4);
				op_block_size = (simd_vec_width*2);
			} else {
				ip_block_size = (simd_vec_width*2);
				op_block_size = (simd_vec_width*4);
			}

			int nInputPlane_block = nInputPlanes/ip_block_size;
			int nOutputPlane_block = nOutputPlanes/op_block_size;

			float *dst = weight_flat;

			for (int dposy=0; dposy<3; dposy++) {
				for (int ii0=0; ii0<nInputPlane_block; ii0++) {
					for (int oi0=0; oi0<nOutputPlane_block; oi0++) {
						for (int dposx=0; dposx<3; dposx++) {
							if (simd_oplane) {
								for (int ii1=0; ii1<ip_block_size; ii1++) {
									for (int oi1=0; oi1<op_block_size; oi1++) {
										int ii = ii0*ip_block_size + ii1;
										int oi = oi0*op_block_size + oi1;
										int mi = oi*nInputPlanes + ii;

										W2Mat &wm = weights[mi];
										float &src = wm.at<float>(dposy, dposx);
										*dst = src;

										dst++;
									}
								}
							} else {
								for (int oi1=0; oi1<op_block_size; oi1++) {
									for (int ii1=0; ii1<ip_block_size; ii1++) {
										int ii = ii0*ip_block_size + ii1;
										int oi = oi0*op_block_size + oi1;
										int mi = oi*nInputPlanes + ii;

										W2Mat &wm = weights[mi];
										float &src = wm.at<float>(dposy, dposx);
										*dst = src;

										dst++;
									}
								}
							}
						}
					}
				}
			}
		} else {
			/* | i0        | i1        | i2 .. iN-1|   i0      | i1        | ..
			 * |o0 o1 o2 o3|o0 o1 o2 o3| ....      |o4 o5 o6 o7|o4 o5 o6 o7| ..
			 * |<-       ->|
			 * | VEC_WIDTH |
			 * |   x  9    |
			 */

			for (int oi=0; oi<nOutputPlanes; oi++) {
				for (int ii=0; ii<nInputPlanes; ii++) {
					int mi = oi*nInputPlanes+ii;
					W2Mat &wm = weights[mi];
					const float *src0 = wm.ptr<float>(0);
					const float *src1 = wm.ptr<float>(1);
					const float *src2 = wm.ptr<float>(2);

					int oi_0 = oi % vec_width;
					int oi_1 = (oi / vec_width) * vec_width;

					float *dst = weight_flat + ((ii*weight_step + oi_1) * 9) + oi_0;
					dst[0*vec_width] = src0[0];
					dst[1*vec_width] = src0[1];
					dst[2*vec_width] = src0[2];

					dst[3*vec_width] = src1[0];
					dst[4*vec_width] = src1[1];
					dst[5*vec_width] = src1[2];

					dst[6*vec_width] = src2[0];
					dst[7*vec_width] = src2[1];
					dst[8*vec_width] = src2[2];
				}
			}
		}
	}

	bool compare_result = false;

#ifdef COMPARE_RESULT
	compare_result = true;
#endif

	size_t in_size = size.width * size.height * sizeof(float) * nInputPlanes;
	size_t out_size = size.width * size.height * sizeof(float) * nOutputPlanes;

	if (compare_result) {
		Buffer *packed_output_cv_buf = new Buffer(env, sizeof(float) * size.width * size.height * nOutputPlanes);

		double t0 = getsec();
		filter_CPU(env, packed_input_buf, packed_output_cv_buf, size);
		//filter_FMA_impl(packed_input, packed_output_cv,
		//		nInputPlanes, nOutputPlanes, fbiases_flat, weight_flat, size, nJob);
		double t1 = getsec();

		/* 3x3 = 9 fma */
		double ops = size.width * size.height * 9.0 * 2.0 * nOutputPlanes * nInputPlanes;

		if (proc->type == W2XCONV_PROC_OPENCL) {
			filter_OpenCL_impl(env, packed_input_buf, packed_output_buf,
					   nInputPlanes, nOutputPlanes, fbiases_flat, weight_flat,
					   size.width, size.height, nJob);
		} else if (proc->type == W2XCONV_PROC_CUDA) {
			filter_CUDA_impl(env, packed_input_buf, packed_output_buf,
					 nInputPlanes, nOutputPlanes, fbiases_flat, weight_flat,
					 size.width, size.height, nJob);
		} else {
			const float *packed_input = (float*)packed_input_buf->get_read_ptr_host(env, in_size);
			float *packed_output = (float*)packed_output_buf->get_write_ptr_host(env);

			switch (proc->sub_type) {
#ifdef X86OPT
			case W2XCONV_PROC_HOST_FMA:
				filter_FMA_impl(env, packed_input, packed_output,
						nInputPlanes, nOutputPlanes, fbiases_flat, weight_flat,
						size.width, size.height, nJob);
				break;

			case W2XCONV_PROC_HOST_AVX:
				filter_AVX_impl(env, packed_input, packed_output,
						nInputPlanes, nOutputPlanes, fbiases_flat, weight_flat,
						size.width, size.height, nJob);
				break;

			case W2XCONV_PROC_HOST_SSE3:
				filter_SSE_impl(env, packed_input, packed_output,
						nInputPlanes, nOutputPlanes, fbiases_flat, weight_flat,
						size.width, size.height, nJob);
				break;
#endif
#ifdef ARMOPT
			case W2XCONV_PROC_HOST_NEON:
				filter_NEON_impl(env, packed_input, packed_output,
						nInputPlanes, nOutputPlanes, fbiases_flat, weight_flat,
						size.width, size.height, nJob);
				break;
#endif

			default:
				filter_CPU(env, packed_input_buf, packed_output_buf, size);
				break;
			}
		}

		double t2 = getsec();

		printf("(w=%d,h=%d) (ip=%d,op=%d) %f %f %f[gflops]\n", size.width, size.height, nInputPlanes, nOutputPlanes, t1-t0, t2-t1, ops/(1000*1000*1000));
		printf("ver2 : %f [Gflops]\n", (ops/(1000.0*1000.0*1000.0)) / (t2-t1));
		printf("orig : %f [Gflops]\n", (ops/(1000.0*1000.0*1000.0)) / (t1-t0));
		int error_count = 0;

		float *packed_output_cv = (float*)packed_output_cv_buf->get_read_ptr_host(env, out_size);
		float *packed_output = (float*)packed_output_buf->get_read_ptr_host(env, out_size);

		for (int i=0; i<size.width * size.height * nOutputPlanes; i++) {
			float v0 = packed_output_cv[i];
			float v1 = packed_output[i];
			float d = fabs(v0 - v1);

			float r0 = d/fabs(v0);
			float r1 = d/fabs(v1);

			float r = (std::max)(r0, r1);

			if (r > 0.1f && d > 0.000001f) {
				int plane = i % nOutputPlanes;
				int pixpos = i / nOutputPlanes;
				int xpos = pixpos % size.width;
				int ypos = pixpos / size.width;

				printf("d=%.20f %.20f %.20f @ (%d,%d,%d,%d) \n",r, v0, v1, xpos, ypos, plane, i);
				error_count++;

				if (error_count >= 256) {
					exit(1);
				}
			}
		}

		if (error_count != 0) {
			exit(1);
		}

		delete packed_output_cv_buf;
	} else {
		if (proc->type == W2XCONV_PROC_OPENCL) {
			filter_OpenCL_impl(env, packed_input_buf, packed_output_buf,
					   nInputPlanes, nOutputPlanes, fbiases_flat, weight_flat,
					   size.width, size.height, nJob);
		} else if (proc->type == W2XCONV_PROC_CUDA) {
			filter_CUDA_impl(env, packed_input_buf, packed_output_buf,
					 nInputPlanes, nOutputPlanes, fbiases_flat, weight_flat,
					 size.width, size.height, nJob);
		} else {
			const float *packed_input = (float*)packed_input_buf->get_read_ptr_host(env, in_size);
			float *packed_output = (float*)packed_output_buf->get_write_ptr_host(env);

			switch (proc->sub_type) {
#ifdef X86OPT
			case W2XCONV_PROC_HOST_FMA:
				filter_FMA_impl(env, packed_input, packed_output,
						nInputPlanes, nOutputPlanes, fbiases_flat, weight_flat,
						size.width, size.height, nJob);
				break;

			case W2XCONV_PROC_HOST_AVX:
				filter_AVX_impl(env, packed_input, packed_output,
						nInputPlanes, nOutputPlanes, fbiases_flat, weight_flat,
						size.width, size.height, nJob);
				break;

			case W2XCONV_PROC_HOST_SSE3:
				filter_SSE_impl(env, packed_input, packed_output,
						nInputPlanes, nOutputPlanes, fbiases_flat, weight_flat,
						size.width, size.height, nJob);
				break;
#endif
#ifdef ARMOPT
			case W2XCONV_PROC_HOST_NEON:
				filter_NEON_impl(env, packed_input, packed_output,
						 nInputPlanes, nOutputPlanes, fbiases_flat, weight_flat,
						 size.width, size.height, nJob);
				break;
#endif
			default:
				filter_CPU(env, packed_input_buf, packed_output_buf, size);
				break;
			}
		}
	}

	w2xc_aligned_free(fbiases_flat);
	w2xc_aligned_free(weight_flat);

	return true;

}

bool Model::filter(W2XConv *conv,
		   ComputeEnv *env,
		   Buffer *packed_input_buf,
		   Buffer *packed_output_buf,
		   W2Size const &size)
{
	bool ret;

	bool avx_available = true;
	bool cl_available = true;
	bool cuda_available = true;

	if (nOutputPlanes > GPU_VEC_WIDTH) {
		cl_available = false;
		cuda_available = false;
	}

	if (nOutputPlanes == 32 && nInputPlanes == 1) {
		/* i1 o32 filter */
	} else if (nOutputPlanes == 1 && nInputPlanes == 128) {
		/* i128 o32 filter */
	} else if (nOutputPlanes == 32 && nInputPlanes == 3) {
		/* i3 o32 filter */
	} else if (nOutputPlanes == 3 && nInputPlanes == 128) {
		/* i128 o3 filter */
	} else if (nInputPlanes == 256) {
		cl_available = false;
		cuda_available = false;
		avx_available = false;
	} else {
		if (nInputPlanes & 1) {
			cl_available = false;
			cuda_available = false;
			avx_available = false;
		}

		if (nOutputPlanes & 31) {
			cl_available = false;
			cuda_available = false;
			avx_available = false;
		}

		if (nInputPlanes == 32 || nInputPlanes == 64 || nInputPlanes == 128) {
			/* ok */
		} else {
			cuda_available = false;
		}
	}

	//printf("[%d->%d] CUDA:%d CL:%d AVX:%d\n", nInputPlanes, nOutputPlanes,
	//       (int)cuda_available,
	//       (int)cl_available,
	//       (int)avx_available);

	const struct W2XConvProcessor *proc = conv->target_processor;
	if ((cl_available && proc->type == W2XCONV_PROC_OPENCL) ||
	    (cuda_available && proc->type == W2XCONV_PROC_CUDA) ||
	    (avx_available && proc->type == W2XCONV_PROC_HOST))
	{
		ret = filter_AVX_OpenCL(conv, env, packed_input_buf, packed_output_buf, size);
	} else {
		ret = filter_CPU(env, packed_input_buf, packed_output_buf, size);
	}

	return ret;
}

bool Model::loadModelFromJSONObject(picojson::object &jsonObj) {

	// nInputPlanes,nOutputPlanes,strideSize,kernelSize,padSize have already set.
	int matProgress = 0;
	picojson::array &wOutputPlane = jsonObj["weight"].get<picojson::array>();

	// setting weight matrices
	for (auto&& wInputPlaneV : wOutputPlane) {
		picojson::array &wInputPlane = wInputPlaneV.get<picojson::array>();

		for (auto&& weightMatV : wInputPlane) {
			picojson::array &weightMat = weightMatV.get<picojson::array>();
			W2Mat writeMatrix(kernelSize, kernelSize, CV_32FC1);

			for (int writingRow = 0; writingRow < kernelSize; writingRow++) {
				auto& weightMatRowV = weightMat.at(writingRow);
				picojson::array &weightMatRow = weightMatRowV.get<
						picojson::array>();

				for (int index = 0; index < kernelSize; index++) {
					writeMatrix.ptr<float>(writingRow)[index] = (float) weightMatRow[index].get<double>();
				} // for(weightMatRow) (writing 1 row finished)

			} // for(weightMat) (writing 1 matrix finished)

			weights.push_back(std::move(writeMatrix));
			matProgress++;
		} // for(wInputPlane) (writing matrices in set of wInputPlane finished)

	} //for(wOutputPlane) (writing all matrices finished)

	// setting biases
	picojson::array biasesData = jsonObj["bias"].get<picojson::array>();
	for (int index = 0; index < nOutputPlanes; index++) {
		biases[index] = biasesData[index].get<double>();
	}

	return true;
}

modelUtility * modelUtility::instance = nullptr;

modelUtility& modelUtility::getInstance(){
	if(instance == nullptr){
		instance = new modelUtility();
	}
	return *instance;
}

Model::Model(FILE *binfp)
{
	uint32_t nInputPlanes, nOutputPlanes, strideSize, kernelSize, padSize;

	fread(&nInputPlanes, 4, 1, binfp);
	fread(&nOutputPlanes, 4, 1, binfp);

	fread(&strideSize, 4, 1, binfp);
	fread(&kernelSize, 4, 1, binfp);
	fread(&padSize, 4, 1, binfp);

	this->nInputPlanes = nInputPlanes;
	this->nOutputPlanes = nOutputPlanes;
	this->strideSize = strideSize;
	this->kernelSize = kernelSize;
	this->padSize = padSize;
	this->weights.clear();
	this->biases.clear();

	// setting weight matrices
	for (uint32_t oi=0; oi<nOutputPlanes; oi++) {
		for (uint32_t ii=0; ii<nInputPlanes; ii++) {
			W2Mat writeMatrix(kernelSize, kernelSize, CV_32FC1);
			for (uint32_t yi=0; yi<kernelSize; yi++) {
				for (uint32_t xi=0; xi<kernelSize; xi++) {
					double v;
					fread(&v, 8, 1, binfp);
					writeMatrix.at<float>(yi, xi) = (float) v;
				}
			}
			this->weights.emplace_back(std::move(writeMatrix));
		}
	}

	for (uint32_t oi=0; oi<nOutputPlanes; oi++) {
		double v;
		fread(&v, 8, 1, binfp);
		biases.push_back(v);
	}
}

Model::Model(int nInputPlane,
	     int nOutputPlane,
	     const float *coef_list,
	     const float *bias)
{
	this->nInputPlanes = nInputPlane;
	this->nOutputPlanes = nOutputPlane;
	this->strideSize = 1;
	this->kernelSize = 3;
	this->padSize = 0;
	this->weights.clear();
	this->biases.clear();

	int cur = 0;
	// setting weight matrices
	for (uint32_t oi=0; oi<(uint32_t)nOutputPlanes; oi++) {
		for (uint32_t ii=0; ii<(uint32_t)nInputPlanes; ii++) {
			W2Mat writeMatrix(kernelSize, kernelSize, CV_32FC1);
			for (int yi=0; yi<kernelSize; yi++) {
				for (int xi=0; xi<kernelSize; xi++) {
					double v = coef_list[cur++];
					writeMatrix.at<float>(yi, xi) = (float) v;
				}
			}
			this->weights.emplace_back(std::move(writeMatrix));
		}
	}

	for (uint32_t oi=0; oi<(uint32_t)nOutputPlanes; oi++) {
		double v = bias[oi];
		biases.push_back(v);
	}
}



bool modelUtility::generateModelFromJSON(const std::string &fileName,
		std::vector<std::unique_ptr<Model> > &models) {

	std::string binpath = fileName + ".bin";
	FILE *binfp = fopen(binpath.c_str(), "rb");

	if (binfp) {
		bool need_update = update_test(binpath.c_str(), fileName.c_str());

		if (need_update) {
			fclose(binfp);
			binfp = NULL;
		}
	}

	if (binfp) {
		uint32_t nModel;

		fread(&nModel, 4, 1, binfp);

		for (uint32_t i=0; i<nModel; i++) {
			std::unique_ptr<Model> m = std::unique_ptr<Model>(
				new Model(binfp));
			models.push_back(std::move(m));
		}

		fclose(binfp);
	} else {
		std::ifstream jsonFile;

		jsonFile.open(fileName);
		if (!jsonFile.is_open()) {
			std::cerr << "Error : couldn't open " << fileName << std::endl;
			return false;
		}

		picojson::value jsonValue;
		jsonFile >> jsonValue;

		std::string errMsg = picojson::get_last_error();
		if (!errMsg.empty()) {
			std::cerr << "Error : PicoJSON Error : " << errMsg << std::endl;
			return false;
		}

		picojson::array& objectArray = jsonValue.get<picojson::array>();
		for (auto&& obj : objectArray) {
			std::unique_ptr<Model> m = std::unique_ptr<Model>(
				new Model(obj.get<picojson::object>()));
			models.push_back(std::move(m));
		}

		binfp = fopen(binpath.c_str(), "wb");
		if (binfp) {
			size_t nModel = objectArray.size();

			fwrite(&nModel, 4, 1, binfp);
			for (auto&& m : models) {
				uint32_t nInputPlanes = m->getNInputPlanes();
				uint32_t nOutputPlanes = m->getNOutputPlanes();
				uint32_t strideSize = m->getStrideSize();
				uint32_t kernelSize = m->getKernelSize();
				uint32_t padSize = m->getPadSize();

				fwrite(&nInputPlanes, 4, 1, binfp);
				fwrite(&nOutputPlanes, 4, 1, binfp);

				fwrite(&strideSize, 4, 1, binfp);
				fwrite(&kernelSize, 4, 1, binfp);
				fwrite(&padSize, 4, 1, binfp);

				std::vector<W2Mat> &weights = m->getWeigts();

				int nw = (int) weights.size();
				for (int wi=0; wi<nw; wi++) {
					W2Mat &wm = weights[wi];
					double v;
					v = wm.at<float>(0,0);
					fwrite(&v, 1, 8, binfp);
					v = wm.at<float>(0,1);
					fwrite(&v, 1, 8, binfp);
					v = wm.at<float>(0,2);
					fwrite(&v, 1, 8, binfp);

					v = wm.at<float>(1,0);
					fwrite(&v, 1, 8, binfp);
					v = wm.at<float>(1,1);
					fwrite(&v, 1, 8, binfp);
					v = wm.at<float>(1,2);
					fwrite(&v, 1, 8, binfp);

					v = wm.at<float>(2,0);
					fwrite(&v, 1, 8, binfp);
					v = wm.at<float>(2,1);
					fwrite(&v, 1, 8, binfp);
					v = wm.at<float>(2,2);
					fwrite(&v, 1, 8, binfp);
				}

				std::vector<double> &b = m->getBiases();
				fwrite(&b[0], 8, b.size(), binfp);
			}

			fclose(binfp);
		}
	}

	return true;
}

void
modelUtility::generateModelFromMEM(int layer_depth,
				   int num_input_plane,
				   const int *num_map, // num_map[layer_depth]
				   const float *coef_list, // coef_list[layer_depth][num_map][kernelSizexkernelSize]
				   const float *bias, // bias[layer_depth][num_map]
				   std::vector<std::unique_ptr<Model> > &models,
				   int kernelSize
	)
{
	int cur = 0;
	models.resize(layer_depth);

	models[0] = std::unique_ptr<Model>(new Model(num_input_plane,
						     num_map[0],
						     &coef_list[0],
						     &bias[0]));

	cur += num_map[0];

	for (int li=1; li<layer_depth; li++) {
		models[li] = std::unique_ptr<Model>(new Model(num_map[li-1],
							      num_map[li],
							      &coef_list[cur * kernelSize * kernelSize],
							      &bias[cur]));

		cur += num_map[li];
	}
}



bool modelUtility::setNumberOfJobs(int setNJob){
	if(setNJob < 1)return false;
	nJob = setNJob;

	return true;
};

int modelUtility::getNumberOfJobs(){
	return nJob;
}

// for debugging

void Model::printWeightMatrix() {

	for (auto&& weightMatrix : weights) {
		//std::cout << weightMatrix << std::endl;
	}

}

void Model::printBiases() {

	for (auto&& bias : biases) {
		std::cout << bias << std::endl;
	}
}

}
