// *********************************************************************

#include "ig-aux.h"
#include "tuplasg.h"
#include "lector-ply.h"
#include "matrices-tr.h"
#include "examen.h"
#include "latapeones.h"
#include <cmath>

using namespace std ;

// *****************************************************************************

MallaCil::MallaCil(const int n){


	for(int i=0; i<=n; i++){

		const float f=float(i)/float(n),
		ang=2.0*M_PI*f,
		vx=std::cos(ang),
		vz=std::sin(ang);

		//Vertice 3*i(lateral del cilindro,inferior)
		vertices.push_back({vx,0.0,vz});
		nor_ver.push_back({vx,0.0,vz});
		cc_tt_ver.push_back({-f,1});

		//Vertice 3*i+1(lateral del cilindro,superior)
		vertices.push_back({vx,1.0,vz});
		nor_ver.push_back({vx,0.0,vz});
		cc_tt_ver.push_back({-f,0});

		//vertice 3*i+2(tapa, es el anterior vertice duplicado)
		vertices.push_back({vx,1.0,vz});
		nor_ver.push_back({0.0,1.0,0.0});
		cc_tt_ver.push_back({(1+vx)*0.5,0.5*(1+vz)});//Revisar

		//triangulos
		if(i<n){
			triangulos.push_back({3*i,3*i+1,3*(i+1)});
			triangulos.push_back({3*(i+1),3*i+1,3*(i+1)+1});
			triangulos.push_back({3*i+2,3*(n+1),3*(i+1)+2});
		}
}
	//vertice 3*(n+1)(centro de la tapa superior)
	vertices.push_back({0.0,1.0,0.0});
	nor_ver.push_back({0,1,0});
	cc_tt_ver.push_back({0.5,0.5});

}

// *****************************************************************************
NodoCil::NodoCil(){
  ponerNombre("Cilindro UGR");

  Textura* t_ugr=new Textura("../recursos/imgs/window-icon.jpg");
  Material* m_ugr=new Material(t_ugr,0.5,0.2,0.6,25.0);

  ponerColor({0.4,0.4,0.4});

  agregar(MAT_Escalado(1,2.5,1));
  agregar(m_ugr);
  agregar(new MallaCil(30));

}

// *****************************************************************************
VariosCil::VariosCil(int n){
	ponerNombre("VariosCil");

	//Pongo  las varias latas LataPeones
	agregar(new VariasLatasPeones());

	//Traslado el círculo para que no se me pise con VariasLatasPeones
	agregar (MAT_Traslacion(n*2,0,0));
	//agregar (MAT_Escalado());

	//Creo el círculo donde pongo los cilindros
  for(int i = 0; i < n; i++){
    float alpha = i*2.0*M_PI/n;

		agregar(MAT_Traslacion(n*cos(alpha),0,n*sin(alpha)));
		NodoCil * nodo=new NodoCil();

		//Los identificadores los pongo a partir del 9000000
		nodo->ponerIdentificador(9000000+i);

		//Solo falta escalar cada uno de los nodos de forma que cuando los dibujemos no se superpongan unos con los otros para ellos debemos realizar una matriz de escalado
		//nodo->agregar(MAT_Escalado());
		agregar(nodo);

  }
	}
