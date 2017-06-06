#include "opencv2/opencv.hpp"
#include <iostream>
#include <fstream>

#define Debug
#include "Scena.h"
using namespace cv;
using namespace std;
vector<Vec<int, 6>> wyniki;
#ifdef Debug
Mat obr;
#endif
int main(int, char)
{
	fstream plik;
	plik.open("nazwy_zdjec\\nazwy_zdjec.txt", ios::in);
	if (!plik.good()){
		cout << "Nie uda³o siê otworzyæ pliku z nazwami" << endl;
		system("pause");
		return 0;
	}
	std::string nazwaZdjecia;
	getline(plik, nazwaZdjecia);
	int nrSceny=0, ujecie=0;
	Scena* scena = new Scena();
	while (!nazwaZdjecia.empty()){
		if (nrSceny != atoi(nazwaZdjecia.substr(6, 3).c_str())){
			if (nrSceny > 0){
				int wynik[6];
				scena->Wynik(wynik);
				wyniki.emplace_back(wynik);
			}
			nrSceny = atoi(nazwaZdjecia.substr(6, 3).c_str());
			delete scena;
			scena = new Scena();
		}
		cout << nazwaZdjecia;
		scena->dodajUjecie(String("zdjecia\\")+nazwaZdjecia);
		getline(plik, nazwaZdjecia);
	}
	int twynik[6];
	scena->Wynik(twynik);
	wyniki.emplace_back(twynik);
	ofstream wynik("wyniki\\Kamil_Urbaniak.txt");
	for (int i = 0; i < nrSceny; i++)
	{
		string linia;
		for (int j = 0; j < 6; j++)
		{
			linia += to_string(wyniki[i][j]);
			if (j < 5)
				linia += ", ";
		}
		linia += 'CR' + 'LF';
		wynik << linia << endl;
	}
	wynik.close();
	return 0;
}