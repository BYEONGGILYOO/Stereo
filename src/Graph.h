#include <iostream>
#include <vector>
#include "openCV.h"

class Vertex;
typedef struct _edge_dist
{
	cv::Mat R, T;
}EdgeDist;
typedef struct _stereo_database
{
	std::vector<cv::KeyPoint> m_KeyPoint;			// keypoint
	cv::Mat m_Descriptor;							// Descriptor
	std::vector<cv::Vec3f> m_WorldCoord;			// 3D coordinates
}Data;

class Edge
{
public:
	Edge(Vertex* src, Vertex* dst, EdgeDist dist)
		:src(src), dst(dst), dist(dist){}
	Vertex* getSrc() { return src; }
	Vertex* getDst() { return dst; }
	EdgeDist getDist() const { return dist; }
private:
	Vertex* src;
	Vertex* dst;
	EdgeDist dist;
};

class Vertex
{
public:
	Vertex(int idx, Data data):idx(idx), data(data)
	{}
	void addEdge(Vertex* v, EdgeDist dist)
	{
		Edge newEdge(this, v, dist);
		edges.push_back(newEdge);
	}
	void printEdge()
	{
		std::cout << "[" << idx << "]" << std::endl;
	}
	std::vector<Edge> getEdges() const { return edges; }
	int getIdx() const { return idx; }
	Data getData() const { return data; }
	void setEdges(const std::vector<Edge> edge) { edges = edge; }
private:
	std::vector<Edge> edges;
	int idx;
	Data data;
};