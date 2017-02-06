#pragma once

#include <windows.h>
#include <sys/timeb.h>
#include <mutex>
#include <map>

//#include <opencv2\opencv.hpp>
//#ifdef _DEBUG
//#pragma comment(lib, "opencv_ts300d.lib")
//#pragma comment(lib, "opencv_world300d.lib")
//#else
//#pragma comment(lib, "opencv_ts300.lib")
//#pragma comment(lib, "opencv_world300.lib")
//#endif

struct function_cummunication
{
	/////////////////////////input////////////////////////////////

	// ��ǥ FPS. 
	int target_hz = 30;

	/////////////////////////output////////////////////////////////

	//exit �Լ��� �ݵɶ� ���� �ð��� üũ
	_timeb last_call;

	//exit �� �ٽ� ȣ��Ǵµ� �ɸ��� �ð��� ����
	int ms = 0;

	//(ms + sleep time) �̿��ؼ� FPS �� ���
	//ms �� �ƹ��� ���� target_hz ��ŭ �����ȴ�.
	float actual_hz = 0;

	//navi engine �� �Լ��� �Ѱ� �� �� ����. 
	//Ÿ�Ƿ� ������ �Լ����� �ñ׳�:1, ���Ƿ� �����Ǵ� �ñ׳�:2
	int trigger = 0; // 1:triggering by eng, 2:triggering by itself

					 //�Լ� ���� �ñ׳��� �޾Ҵ��� üũ��. 
					 //ex) trigger = 1, fire = 0 -> ������ ���� �ñ׳��� ������ �Լ� ���� �ȵ�.
					 //ex) trigger = 1, fire = 1 -> ������ ���� �ñ׳��� ���, �Լ� ���� ��.
	int fire = 0;
};


class IPC
{
private:
	struct function
	{
		function_cummunication *fc;

		int MMF_state = 0; // -1: disable, 0 : disconnected, 1: connected, 2: virtual MMF
		int data_size = 0;
		std::string exe_name;
		HANDLE hMapFile;
		LPCTSTR pBuf;
	};

	std::map<std::string, function> f_list;
	std::string own_name;

	int openMMF(std::map<std::string, function>::iterator &_func)
	{
		int data_size = (int)sizeof(function_cummunication) + _func->second.data_size;

		_func->second.hMapFile = CreateFileMapping(
			INVALID_HANDLE_VALUE,    // use paging file
			NULL,                    // default security
			PAGE_READWRITE,          // read/write access
			0,                       // maximum object size (high-order DWORD)
			data_size,                // maximum object size (low-order DWORD)
			_func->first.c_str());                 // name of mapping object

		if (_func->second.hMapFile == NULL)
		{
			printf("Could not create file mapping object (%s).\n", _func->first.c_str());
			_func->second.MMF_state = 0;
			return 0;
		}
		return 1;
	};
public:
	IPC::IPC(std::string _own_name)
	{
		own_name = _own_name;
	};
	IPC::~IPC()
	{
		std::map<std::string, function>::iterator finder = f_list.find(own_name);

		if (finder->second.MMF_state > 0)
		{
			finder->second.fc->trigger = 0;
			finder->second.fc->fire = 0;

			UnmapViewOfFile(finder->second.pBuf);
			CloseHandle(finder->second.hMapFile);
		}
	};

	///Ư�� ���μ����� ������ ����� data type �� �����ִ� ��� �����͸� �����Ŵ
	int connect(std::string f_name, int _data_size = 0)
	{
		int openMMF_flag = 0;
		std::map<std::string, function>::iterator finder = f_list.find(f_name);
		if (finder != f_list.end())
		{
			if (finder->second.MMF_state == 1) return 1;
		}
		else
		{
			f_list.insert(std::pair<std::string, function>(f_name, function()));
			finder = f_list.find(f_name);
		}

		finder->second.data_size = _data_size;
		int data_size = (int)sizeof(function_cummunication) + finder->second.data_size;

		finder->second.exe_name = f_name;

		finder->second.hMapFile = OpenFileMapping(
			FILE_MAP_ALL_ACCESS,   // read/write access
			FALSE,                 // do not inherit the name
			finder->second.exe_name.c_str());

		if (finder->second.hMapFile == NULL)
		{
			printf("Could not open file mapping object (%s).\n", finder->second.exe_name.c_str());
			finder->second.MMF_state = -1;

			if (f_name == own_name)
			{
				printf("Try to create MMF\n");
				if (openMMF(finder) == 0) return 0;
				else openMMF_flag = 1;
			}
			else
			{
				return 0;
			}

		}

		finder->second.pBuf = (LPTSTR)MapViewOfFile(finder->second.hMapFile, // handle to map object
			FILE_MAP_ALL_ACCESS,  // read/write permission
			0,
			0,
			data_size);

		if (finder->second.pBuf == NULL)
		{
			DWORD dw = GetLastError();
			TCHAR* message = nullptr;
			FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_ALLOCATE_BUFFER,
				nullptr,
				dw,
				MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
				(TCHAR *)&message,
				0,
				nullptr);

			printf("Could not map view of file (%s).\n", finder->second.exe_name.c_str());
			printf("%s\n", message);
			CloseHandle(finder->second.hMapFile);
			finder->second.MMF_state = -1;
			return 0;
		}

		finder->second.MMF_state = 1;
		if (openMMF_flag == 1)
		{// initialize itself
			function_cummunication temp;
			std::memcpy((void *)finder->second.pBuf, &temp, sizeof(function_cummunication));
			finder->second.MMF_state = 2;
		}

		finder->second.fc = (function_cummunication*)finder->second.pBuf;
		_ftime64_s(&finder->second.fc->last_call);
		return 1;
	};
	template <typename T> T* connect(std::string f_name)
	{
		int customed_data_size = (int)sizeof(T);
		if (connect(f_name, customed_data_size) == 0) return nullptr;

		std::map<std::string, function>::iterator finder = f_list.find(f_name);

		return (T*)&finder->second.pBuf[sizeof(function_cummunication)];
	};
	template <typename T> T* get_data(std::string f_name)
	{
		std::map<std::string, function>::iterator finder = f_list.find(_exe_name);
		if (finder == f_list.end()) return nullptr;

		std::map<std::string, function>::iterator finder = f_list.find(f_name);
		return (T*)&finder->second.pBuf[sizeof(function_cummunication)];
	};
	function_cummunication* get_state(std::string _exe_name)
	{
		std::map<std::string, function>::iterator finder = f_list.find(_exe_name);
		if (finder == f_list.end()) return nullptr;

		return finder->second.fc;
	};
	int isOn(std::string _exe_name)
	{
		function_cummunication *state = get_state(_exe_name);
		if (state == nullptr) return 0;
		else if (state->fire > 0) return 1;
		else return 0;
	}
	int isConnected(std::string f_name)
	{
		std::map<std::string, function>::iterator finder = f_list.find(f_name);
		if (finder == f_list.end()) return 0;

		if (finder->second.MMF_state == 1) return 1;
		else return 0;
	};

	///Ư�� ���μ��� ������ engine �� ��û��
	int start(std::string _exe_name)
	{
		std::map<std::string, function>::iterator finder = f_list.find(_exe_name);
		if (finder == f_list.end()) return 0;
		if (finder->second.MMF_state <= 0) return 0;

		if (_exe_name == own_name)
		{
			finder->second.fc->fire = 1;
			if (finder->second.fc->trigger == 0) finder->second.fc->trigger = 2;
		}
		else
		{
			finder->second.fc->trigger = 1;
		}

		return 1;
	};

	///Ư�� ���μ��� ������ engine �� ��û��
	int stop(std::string _exe_name)
	{
		std::map<std::string, function>::iterator finder = f_list.find(_exe_name);
		if (finder == f_list.end()) return 0;
		if (finder->second.MMF_state <= 0) return 0;

		finder->second.fc->trigger = 0;
		return 1;
	};

	/// ������ main loop �ȿ� �ѹ� ȣ��Ǿ�� ��. 
	///engine �� ���μ��� ���¸� �����ϰ� �����Ҽ� �ְ���.
	int exit(void)
	{
		std::map<std::string, function>::iterator finder = f_list.find(own_name);
		if (finder == f_list.end()) return 0;
		if (finder->second.MMF_state <= 0) return 0;

		if (finder->second.fc->trigger == 0)
		{
			finder->second.fc->fire = finder->second.fc->trigger;

			finder->second.MMF_state = 0;
			UnmapViewOfFile(finder->second.pBuf);
			CloseHandle(finder->second.hMapFile);
			return 1;
		}

		struct _timeb now_T;
		_ftime64_s(&now_T);
		int sec_gap = (int)(now_T.time - finder->second.fc->last_call.time);
		int mili_gap = (int)(now_T.millitm - finder->second.fc->last_call.millitm);
		float t_gap = (float)(1000 * sec_gap + mili_gap);
		finder->second.fc->ms = (int)t_gap;

		float designed_t_gap = 1000.0f / (float)finder->second.fc->target_hz;
		float laft_t_gap = designed_t_gap - t_gap;
		if (laft_t_gap>0 && t_gap >= 0) Sleep((DWORD)laft_t_gap);

		struct _timeb after_sleep;
		_ftime64_s(&after_sleep);
		sec_gap = (int)(after_sleep.time - finder->second.fc->last_call.time);
		mili_gap = (int)(after_sleep.millitm - finder->second.fc->last_call.millitm);
		t_gap = (float)(1000 * sec_gap + mili_gap);
		finder->second.fc->actual_hz = 1000.0f / t_gap;

		_ftime64_s(&finder->second.fc->last_call);

		return 0;
	};
};


struct Engine
{
	int state = 0;
};

struct Cam_640480
{
	bool imshow = true;
	unsigned char data[640 * 480 * 3];
};

struct Robot
{
	//////////////////////////// input ////////////////////////////
	double set_vel = 0.0;
	double set_rotvel = 0.0;

	bool navigate_on = false;
	bool near_goal = false;
	//double sub_goal_x = 0.0;
	//double sub_goal_y = 0.0;
	//double sub_goal_th = 0.0;
	int sub_goal_number = 0;
	double sub_goal[20][3];

	//////////////////////////// output ////////////////////////////

	int is_on = 0;

	double x = 0;
	double y = 0;
	double th = 0;

	double get_vel = 0.0;
	double get_rotvel = 0.0;
	bool arrived = false;

	double KeyFrame[3][3] = {}; // [idx][val] -> [val] : {Key, t, r}
};

struct ObservationData
{
	//////////////////////////// input ////////////////////////////
	int mode = 0; // 0:localize, 1:save, 2:saveDB, 3:loadDB, 9:reset
	bool always_on_flag = 0;
	bool draw_flag = 0;
	int row = 480;
	int col = 640;
	unsigned char RGB_data[640 * 480 * 3];

	//////////////////////////// output ////////////////////////////
	int num_detected_feature = 0;
	int num_DB_size = 0;
	float score[2048];
	float likelihood[2048];
};

struct Kinect1Data
{
	//////////////////////////// input ////////////////////////////
	bool get_color = 1;
	bool get_grid = 1;

	int        cgridWidth = 500;
	int        cgridHeight = 500;
	int		   cRobotCol = 250;
	int		   cRobotRow = 350;

	double mm2grid = 100.0 / 1000.0;
	int sampling_gap = 20;

	//////////////////////////// output ////////////////////////////
	unsigned char gridData[500 * 500];
	unsigned char freeData[500 * 500];
	unsigned char occupyData[500 * 500];
	unsigned char colorData[2][640 * 480 * 4];
	double RT[2][16];
};
struct Kinect2Data
{
	//////////////////////////// input ////////////////////////////
	double setYaw = 0.0;
	double setPitch = 0.0;
	bool draw_color = 0;

	//////////////////////////// output ////////////////////////////
	int        cColorWidth = 960; // 1920 / 2;
	int        cColorHeight = 540; // 1080 / 2;
	unsigned char data[960 * 540 * 3];
};

struct HumanReIdentificationData
{

	//////////////////////////// input ////////////////////////////
	int Mode = 0;
	bool draw_flag = 1;
	bool TakePicture = 1;
	//////////////////////////// output ////////////////////////////
	float score[5 + 1];
	float Color_likelihood[5 + 1];
	float Texture_likelihood[5 + 1];
	int Color_result = 0;
	int Texture_result = 0;
	int people_count = 0;
};