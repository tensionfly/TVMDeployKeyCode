#include <iostream>
#include <string>
#include <fstream>
#include <typeinfo>
#include <cmath>
#include <tchar.h>
#include <stdio.h>
#include <numeric>

//#include <algorithm>

#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <dlpack/dlpack.h>

#include<opencv2/opencv.hpp>

using namespace cv;
using namespace std;

constexpr int INPUT_IMG_WIDTH = 416;
constexpr int INPUT_IMG_HEIGHT = 416;
constexpr int CLASSES_NUM = 2;
constexpr int ANCHORS_NUM = (13*13+26*26+52*52)*3;
constexpr int ANCHORS[9][2] = { {33, 33}, {60, 60}, {92, 93},
								{119, 42}, {22, 132}, {166, 110},
								{51, 175}, {97, 213}, {180, 216} };
constexpr int ANCHORS_MASK[3][3] = { {6,7,8}, {4,5,6}, {0,1,2} };
constexpr int STRIDE[3] = { 13,26,52 };
const vector<string> CLASS_NAMES ={ "ship", "airplane" };

void cvtMat2MXData(const Mat& img, float* pData, 

	float mean_r = 0.0f, float mean_g = 0.0f, float mean_b = 0.0f) {
	int w = img.cols, h = img.rows;

	CHECK_EQ(w, INPUT_IMG_WIDTH);
	CHECK_EQ(h, INPUT_IMG_HEIGHT);
	int size = w * h;

	float* ptr_im_b = pData, * ptr_im_g = pData + size, * ptr_im_r = pData + size + size;
	for (int i = 0; i < h; ++i) {
		auto ptr = img.ptr<uchar>(i);
		for (int j = 0; j < w; ++j) {
			*ptr_im_b++ = (static_cast<float>(*ptr++) - mean_b)/255.0;
			*ptr_im_g++ = (static_cast<float>(*ptr++) - mean_g)/255.0;
			*ptr_im_r++ = (static_cast<float>(*ptr++) - mean_r)/255.0;
		}
	}
}

void transform(Mat& img, Mat& img_out)
{
	/*int w = img.cols, h = img.rows;
	for (int i = 0;i < h;++i){	
		auto ptr = img.ptr<uchar>(i);
		for (int j = 0;j < w;++j){	
			img_out.data[(i*w+w-j-1)*3] = static_cast<float>(*ptr++);
			img_out.data[(i * w + w - j - 1) * 3+1] = static_cast<float>(*ptr++);
			img_out.data[(i * w + w - j - 1) * 3+2] = static_cast<float>(*ptr++);
		}
	}*/

	/*Mat bgr2rgb;
	cvtColor(img, bgr2rgb, COLOR_BGR2RGB);*/

	resize(img, img_out, Size(INPUT_IMG_WIDTH, INPUT_IMG_HEIGHT));

}

float IouCal(vector<float> box1, vector<float> box2) {
	float ix0 = max(box1[0], box2[0]);
	float iy0 = max(box1[1], box2[1]);
	float ix1 = min(box1[2], box2[2]);
	float iy1 = min(box1[3], box2[3]);

	float area1 = max(0.0f,(box1[2] - box1[0]))* max(0.0f,(box1[3] - box1[1]));
	float area2 = max(0.0f,(box2[2] - box2[0])) * max(0.0f,(box2[3] - box2[1]));
	float inarea = (ix1 - ix0) * (iy1 - iy0);

	return inarea / (area1 + area2 - inarea);
}

template <typename T>
vector<size_t> sort_indexes_e(vector<T>& v)
{
	vector<size_t> idx(v.size());
	iota(idx.begin(), idx.end(), 0);
	sort(idx.begin(), idx.end(),
		[&v](size_t i1, size_t i2) {return v[i1] < v[i2]; });
	return idx;
}

void NMSProcess(vector<vector<float>> input, vector<vector<float>>& output, float nms_thresh) {
	for (int i = 0;i < input.size();++i) {
		auto cls_input = input[i];
		int num = cls_input.size() / 5;
		vector<vector<float>> inputxy(num);
		vector<float> scores(num);
		for (int k = 0;k < num;++k) scores[k] = cls_input[k * 5 + 4];
	
		vector<size_t> idx;
		idx = sort_indexes_e(scores);
		for (int k = 0;k < num;++k) {
			inputxy[k].push_back(cls_input[(idx[num - 1 - k]) * 5]);
			inputxy[k].push_back(cls_input[(idx[num - 1 - k]) * 5 + 1]);
			inputxy[k].push_back(cls_input[(idx[num - 1 - k]) * 5 + 2]);
			inputxy[k].push_back(cls_input[(idx[num - 1 - k]) * 5 + 3]);
			scores[k] = cls_input[(idx[num - 1 - k]) * 5 + 4];
		}
		//inputxy max 2 min 

		vector<int> keep_vec(num, 1);
		for (int j = 0;j < num;++j) {
			auto current_box = inputxy[j];
			for (int m = j + 1;m < num;++m) {
				float iou_value = IouCal(current_box, inputxy[m]);
				if (iou_value > nms_thresh) keep_vec[m] = 0;
			}
		}

		for (int n = 0;n < num;++n) {
			if (keep_vec[n] == 1) {
				output[i].push_back(inputxy[n][0]);
				output[i].push_back(inputxy[n][1]);
				output[i].push_back(inputxy[n][2]);
				output[i].push_back(inputxy[n][3]);
				output[i].push_back(scores[n]);
			}
		}

	}
	cout << output[0].size() << output[1].size() << endl;

}

void WriteResult(Mat& img,vector<vector<float>> result,string save_path) {

	/*Mat rgb2bgr;
	cvtColor(img, rgb2bgr, COLOR_RGB2BGR);*/
	for (int i = 0;i < result.size();++i) {
		auto result_cls = result[i];
		int num = result_cls.size() / 5;
		for (int j = 0;j < num;++j) {
			int x0 = min(max(0,int(result_cls[j*5])),INPUT_IMG_WIDTH);
			int y0 = min(max(0, int(result_cls[j * 5+1])), INPUT_IMG_HEIGHT);
			int x1 = min(max(0, int(result_cls[j * 5+2])), INPUT_IMG_WIDTH);
			int y1 = min(max(0, int(result_cls[j * 5+3])), INPUT_IMG_HEIGHT);
			int w = x1 - x0;
			int h = y1 - y0;
			rectangle(img, Rect(x0, y0, h, w), Scalar(0, 0, 255), 2, 1, 0);
			putText(img, CLASS_NAMES[i]+": "+ to_string(result_cls[j * 5 + 4]), Point(x0, y0), FONT_HERSHEY_PLAIN,1, Scalar(0, 0, 255), 2);
			/*imshow("img", img);
			waitKey(0);*/
		}
	}
	imwrite(save_path, img);
}

void PostProcess(float* result,Mat img_read,const string save_path,float bconf,float nms_thresh) {
	//int w = img_read.cols, h = img_read.rows;
	int cyc_num = 5 + CLASSES_NUM;
	vector<vector<float>> result_cls(CLASSES_NUM);
	int st = 0;

	for (int i = 0; i < 3;++i) {
		int stride = STRIDE[i];
		float scale_h = INPUT_IMG_HEIGHT * 1.0 / stride;
		float scale_w = INPUT_IMG_WIDTH * 1.0 / stride;
		float anchor3[3][2];

		for (int k = 0;k < 3;++k) {
			int index = ANCHORS_MASK[i][k];
			anchor3[k][0] = ANCHORS[index][0];
			anchor3[k][1] = ANCHORS[index][1];
		}

		for (int j =st ;j < st+stride * stride *3;++j) {
			int num = (j - st) / 3;
			int y_index = num / stride;
			int x_index = num % stride;
			int anchor3_index = (j - st) % 3;

			float result_j0 = (result[j*cyc_num]+float(x_index))* scale_w;
			float result_j1 = (result[j * cyc_num + 1]+float(y_index))*scale_h;
			float result_j2 = exp(result[j * cyc_num + 2]) * anchor3[anchor3_index][0];
			float result_j3 = exp(result[j * cyc_num + 3]) * anchor3[anchor3_index][1];

			result[j * cyc_num] = result_j0 - result_j2 / 2;
			result[j * cyc_num + 1] = result_j1 - result_j3 / 2;
			result[j * cyc_num + 2] = result_j0 + result_j2 / 2;
			result[j * cyc_num + 3] = result_j1 + result_j3 / 2;

			//cout << result_j0 <<" "<< result_j1 << " " << result_j2 << " " << result_j3<<endl;
			if (result[j * cyc_num + 4] > bconf) {
				float score = 0.0;
				//cout << "hehe" << endl;
				int max_score_index = 0;
				for (int a = 5;a < cyc_num;++a) {
					if (result[j * cyc_num + a] > score) {
						max_score_index = a;
						score = result[j * cyc_num + a];
					}
				}
				result_cls[max_score_index - 5].push_back(result[j * cyc_num]);
				result_cls[max_score_index - 5].push_back(result[j * cyc_num+1]);
				result_cls[max_score_index - 5].push_back(result[j * cyc_num+2]);
				result_cls[max_score_index - 5].push_back(result[j * cyc_num+3]);
				result_cls[max_score_index - 5].push_back(score);
			}
			
		}
		st+= stride * stride * 3;
	}
	//result_cls [cls,(x0,y0,x1,y1,score)]
	
	//Do NMS
	vector<vector<float>> result_nms(CLASSES_NUM);
	NMSProcess(result_cls, result_nms, nms_thresh);
	WriteResult(img_read, result_nms, save_path);

}

void inferance(const string dll_path, const string json_path, const string params_path, 
	const vector<string> img_path_vector,const string save_dir,const float bconf,const float nms_thresh) {

	//step 1 
	tvm::runtime::Module mod_dylib =
		tvm::runtime::Module::LoadFromFile(dll_path);

	std::ifstream json_in(json_path, std::ios::in);
	std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
	json_in.close();

	std::ifstream params_in(params_path, std::ios::binary);
	std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
	params_in.close();

	//step 2
	TVMByteArray params_arr;
	params_arr.data = params_data.c_str();
	params_arr.size = params_data.length();

	//step 3
	int dtype_code = kDLFloat;
	int dtype_bits = 32;
	int dtype_lanes = 1;
	int device_type = kDLCPU;
	int device_id = 0;

	tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))
		(json_data, mod_dylib, device_type, device_id);

	//step 4
	DLTensor* input_x;
	int in_ndim = 4;
	int64_t in_shape[4] = { 1, 3, INPUT_IMG_HEIGHT, INPUT_IMG_WIDTH };
	TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &input_x);

	DLTensor* output_y;
	int out_ndim = 3;
	int64_t out_shape[3] = { 1,ANCHORS_NUM,5+ CLASSES_NUM };
	TVMArrayAlloc(out_shape, out_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &output_y);

	//step 5
	tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
	load_params(params_arr);

	tvm::runtime::PackedFunc run = mod.GetFunction("run");
	tvm::runtime::PackedFunc get_output = mod.GetFunction("get_output");

	//step 6
	Mat img_read;
	Mat img_trans;
	float* pdata = new float[INPUT_IMG_WIDTH * INPUT_IMG_HEIGHT * 3];
	int num_img_path = img_path_vector.size();

	for (int i = 0; i < num_img_path; ++i) {

		img_read = imread(img_path_vector[i]);
		transform(img_read, img_trans);
		cvtMat2MXData(img_trans, pdata);
		memcpy(input_x->data, pdata, 3 * INPUT_IMG_HEIGHT * INPUT_IMG_WIDTH * sizeof(float));

		tvm::runtime::PackedFunc set_input = mod.GetFunction("set_input");
		set_input("data", input_x);

		run();
		get_output(0, output_y);
		auto result = static_cast<float*>(output_y->data);

		//step 7 
		string save_path = save_dir + "result.png";//basename(img_path_vector[i]);
		PostProcess(result, img_trans, save_path,bconf,nms_thresh);

	}
}

int main()
{
	const string dll_path = "D:/zlf/cpp_project/yolo.dll";
	const string json_path = "D:/zlf/cpp_project/yolo.json";
	const string params_path = "D:/zlf/cpp_project/yolo.params";
	const string save_dir = "D:/zlf/cpp_project/";
	const vector<string> img_path_vector = { "D:/zlf/cpp_project/1.png" };

	inferance(dll_path, json_path, params_path, img_path_vector, save_dir, 0.2, 0.4);
	//system("pause");

	return 0;
}