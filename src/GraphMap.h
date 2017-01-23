#pragma once

#include <iostream>
#include <vector>
#include <map>

#include "openCV.h"

class Data {
	cv::Mat image[2];
	std::vector<cv::KeyPoint> m_vecKeyPoint;		// keypoint
	cv::Mat m_vecDescriptor;						// Descriptor
	std::vector<cv::Vec3f> m_vecWorldCoord;			// 3D coordinates
	cv::Mat R, T;
};

template <class wType>
class GraphMap
{
private:
	class Vertex
	{
	private:
		std::pair<int, int> edge;
		wType data;
		Vertex(std::pair<int, int> edge, wType data)
		{

		}
		~Vertex() {};
	};

	struct Edge
	{
		int src, dest;
		Data data;
	};
	
public:
	GraphMap();
	~GraphMap();
private:
	int m_nSize;
	std::vector<std::pair<int, Data>> adjList[100];
	void addEdge(std::vector<Edge> edges)
	{
		for (int i = 0; i < edges.size(); i++)
		{
			int src = edges[i].src;
			int dest = edges[i].dest;
			Data data = edges[i].data;

			adjList[src].push_back(std::make_pair(dest, data));
		}
	}
	Data
};

