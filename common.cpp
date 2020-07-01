/*
 *  common-2018.h
 *
 *  Created on: 31 gen. 2019
 *  Author: ecesar
 *
 *  Descripció:
 *  Funcions auxiliars per la lectura d'un arxiu de patrons format per un
 *  nombre indeterminat de xifres (0-9) escrites a mà representades en matrius
 *  de 32x32 posicions on cada posició pot ser 0 (no hi ha traç) o 1 (n'hi ha).
 *  Els arxius de patrons contenen un cert nombre d'imatges una rere l'altre,
 *  per cada imatge s'indiquen 32 línies amb 32 columnnes de caràcters 0s i 1s i una
 *  darrera línia amb la xifra representada (espai en blanc + xifra entre 0 i 9).
 *  Aquests arxius es corresponen amb conjunts reals que es fan servir en el mon
 *  del ML i sobre els quals podeu trobar més informació en l'enllaç:
 *  https://archive.ics.uci.edu/ml/index.php
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "common.h"


void readImg(char img[NUMIN], FILE *fd ){

        //char *img;

        //img = (char *)malloc(1024);

	for( int i = 0; i < 32; i++ ){
        	fscanf(fd, "%s\n", &img[i*32]);
		for(int j = 0; j < 32; j++) img[i*32+j] -= '0';
	}

        //return img;
}

int loadPatternSet(char tSet[NUMPAT][NUMIN], char *fname, int trainS){

        //char **tset;
	FILE *fd;
	int c;

	if( (fd = fopen( fname, "rb" )) == NULL) return 0;

        //tset = (char **)malloc(nf*sizeof(char *));
        int error = 0;

        for (int i = 0; i < NUMPAT; i++){
                readImg(tSet[i], fd );
		memset( &Target[i], 0, 10*sizeof(float) );
		fscanf( fd, "%d\n", &c );
		if (trainS) Target[i][c] = 1.0; 
		else Validation[i] = c;
        }
	fclose(fd);

        if (error) return 0;
        return 1;
}

void printImg( char *Img, int x ){

	printf("Pattern:\n");
	for (int i = 0; i < 1024; i++ ){
		printf("%c", Img[i] + '0' );
		if (( i != 0 ) && !( i%32 )) printf("\n");
	}
	printf("\nTarget:\n");
	for (int i = 0; i < 10; i++ ) printf("%f ",Target[x][i]);
	printf("\n");
}
 
