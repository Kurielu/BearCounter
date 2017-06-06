#pragma once
#include "opencv2/opencv.hpp"
#include <iostream>
#define Debug
using namespace std;
using namespace cv;
enum TRYB {greater, even};
typedef double ZNACZNIK;
const size_t dl = 3000;
const ZNACZNIK CIRC = 1.545, CORN = 1.512, RECT = 1.484;
const int CIRC_X = (1412 / 1527.0)*dl, CORN_X = (1468 / 1527.0)*dl, RECT_X = dl;
const int CIRC_Y = CIRC_X / CIRC, CORN_Y = CORN_X / CORN, RECT_Y = RECT_X / RECT;
typedef Vec<uchar, 3> Pixel;
#ifdef Debug
extern Mat obr;
#endif
class Scena
{
private:
	vector<Vec<int,6>>wyniki;
	int liczbaUjec;
	std::string ttt;
	Size rozmiarObrazu;
	ZNACZNIK znacznik;
public:
	void dodajUjecie(std::string scieszka);
	void Wynik(int* t);
	Scena();
	~Scena();
#ifdef Debug
	static int nrzd;
#endif
private:
	vector<Point> znajdzWierzcholki(vector<vector<Point>> kontury);
	vector<vector<Point>> wykryjKsztalty(Mat zdj);
	void rzutuj(Mat &zdj, vector<Point> wierzcholki);
	void minCol(Mat in, Mat out, int treshold);
	void maxCol(Mat in1, Mat in2, Mat &out);
	vector<Point> typOkregi(vector<vector<Point>> kontury);
	vector<Point> typPasy(vector<vector<Point>> approx);
	vector<Point> typRogi(vector<vector<Point>> approx);
	vector<vector<Point>> redukujKontury(vector<vector<Point>> kontury, vector<vector<Point>> approx, int liczba_krawedzi, TRYB tryb);
	Point punktPrzeciecia(vector<Point2f> l1, vector<Point2f> l2);
	int findCenter(vector<Point> points);
	bool jestRogiem(vector<Point> ksztalt, Point center);
	float angleBetween(const Point &v1, const Point &v2);
	vector<Point> bestRect(vector<Point> p, ZNACZNIK znacznik);
	void rectOrder(vector<Point> &rec);
	double wskaznikProstokata(vector<Point> rec, ZNACZNIK znacznik);
	void policzZelki(Mat &zdj, vector<Point> wierzcholki);
	void policzKolorowe(Mat &zdj, Mat &binary, vector<Point> wierzcholki, int &out_liczba);
	void policzPrzezroczyste(Mat &zdj, Mat &binary, vector<Point> wierzcholki, int &out_liczba);
	int kupaZelkow(Mat &zdj, vector<Point> wierzcholki, int nr_konturu, vector<vector<Point> > contours, vector<Vec4i> hierarchy);
	int kupaZelkowbial(Mat &zdj, vector<Point> wierzcholki, int nr_konturu, vector<vector<Point> > contours, vector<Vec4i> hierarchy);
};

void kuwahara1c(Mat in_out, int k);
void maxCol(Mat in1, Mat in2, Mat &out);
void SimplestCB(Mat& in, Mat& out, float percent);
void normalizuj(vector<Point>p, Mat &matNormalizacji, ZNACZNIK z);
double pointsDistance(Point2f p1, Point2f p2);
float minimum_distance(Point v, Point w, Point p);