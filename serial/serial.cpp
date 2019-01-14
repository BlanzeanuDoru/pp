
#include "connected.h"

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cfloat>
#include <climits>
#include <algorithm>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cilk/cilk.h>

#define MAX_SOURCE_SIZE INT_MAX
#define MAX_FEATURE_SIZE 162336
#define MAX_CASCADE_LAYERS 31
#define STD_NORM_CONST 10000
#define MAX_RULES_SIZE 202
#define SAMPLE_SIZE 24
#define SCALE_MULTIPLIER 1.5
#define PEN_WIDTH 3

using namespace std;


typedef struct Window {
    int x,y;
    int h,w;
} Window;

typedef struct Feature {
    int type;
    int x;
    int y;
    int h;
    int w;
} Feature;

vector<Window> face_windows;

int closest_integer(int nr,float x) {
    return nr*x+0.5;
}

bool is_inside(Window w1,Window w2) {
    int iy = w1.x + w1.h/2;
    int ix = w1.y + w1.w/2;
    return ( iy > w2.x && iy < w2.x + w2.h && ix > w2.y && ix < w2.y + w2.h );
}

bool window_order (Window w1,Window w2) {
    return w1.h<w2.h;
}

void compute_integral_image(float *image,float *ii,int N,int M) {
    ii[0]=image[0];
    for (int i=1;i<N;i++) {
        for (int j=1;j<M;j++) {
            ii[i*M+j]=image[i*M+j]+ii[(i-1)*M+j]+ii[i*M+j-1]-ii[(i-1)*M+j-1];
        }
    }
}

float compute_rectangle_sum(float* ii,int x,int y,int h,int w,int w_size,int M,int N) {
    int X=x+h-1;
    int Y=y+w-1;
    float sum=0;

    sum=ii[X*N+Y];
    sum-=x>0?ii[(x-1)*N+Y]:0;
    sum-=y>0?ii[X*N+y-1]:0;
    sum+=min(x,y)>0?ii[(x-1)*N+y-1]:0;

    return sum;

}
float feature_scaling(Feature* real_features,float* image,int x,int y,int w_size,int f_idx,bool remove_mean,int M,int N) {
    float a,f,scale;
    int h,w,or_h,or_w,i,j,abs_i,abs_j,type,i1,j1;

    scale=w_size/(SAMPLE_SIZE);
    float mean=remove_mean?compute_rectangle_sum(image,x,y,w_size,w_size,w_size,M,N)/pow((float)w_size,2):0;

    a=real_features[f_idx].h*real_features[f_idx].w;
    i1=i=closest_integer(real_features[f_idx].x,scale);
    i+=x;
    j1=j=closest_integer(real_features[f_idx].y,scale);
    j+=y;
    h=closest_integer(real_features[f_idx].h,scale);
    w=closest_integer(real_features[f_idx].w,scale);

    if (real_features[f_idx].type==1) {
        h=min(h,w_size-i1);
        w=w%2==0?w:w+1;
        while (w+j1>w_size)
            w-=2;

        float S1=compute_rectangle_sum(image,i,j,h,w/2,w_size,M,N);
        float S2=compute_rectangle_sum(image,i,j+w/2,h,w/2,w_size,M,N);

        f=((S1-S2)*a)/(h*w);
    }
    else if (real_features[f_idx].type==2) {
        h=min(h,w_size-i1);
        w=w%3==0?w:w+3-w%3;
        while (w+j1>w_size)
            w-=3;

        float S1=compute_rectangle_sum(image,i,j,h,w/3,w_size,M,N);
        float S2=compute_rectangle_sum(image,i,j+w/3,h,w/3,w_size,M,N);
        float S3=compute_rectangle_sum(image,i,j+2*(w/3),h,w/3,w_size,M,N);

        f=((S1-S2+S3-mean*h*(w/3))*a)/(h*w);
    }
    else if (real_features[f_idx].type==3) {
        w=min(w,w_size-j1);
        h=h%2==0?h:h+1;
        while (h+i1>w_size)
            h-=2;

        float S1=compute_rectangle_sum(image,i,j,h/2,w,w_size,M,N);
        float S2=compute_rectangle_sum(image,i+h/2,j,h/2,w,w_size,M,N);

        f=((S1-S2)*a)/(h*w);
    }
    else if (real_features[f_idx].type==4) {
        w=min(w,w_size-j1);
        h=h%3==0?h:h+3-h%3;
        while (h+i1>w_size)
            h-=3;

        float S1=compute_rectangle_sum(image,i,j,h/3,w,w_size,M,N);
        float S2=compute_rectangle_sum(image,i+h/3,j,h/3,w,w_size,M,N);
        float S3=compute_rectangle_sum(image,i+2*(h/3),j,h/3,w,w_size,M,N);

        f=((S1-S2+S3-mean*(h/3)*w)*a)/(h*w);
    }
    else if (real_features[f_idx].type==5) {
        h=h%2==0?h:h+1;
        w=w%2==0?w:w+1;
        while (h+i1>w_size)
            h-=2;
        while (w+j1>w_size)
            w-=2;

        float S1=compute_rectangle_sum(image,i,j,h/2,w/2,w_size,M,N);
        float S2=compute_rectangle_sum(image,i+h/2,j,h/2,w/2,w_size,M,N);
        float S3=compute_rectangle_sum(image,i,j+w/2,h/2,w/2,w_size,M,N);
        float S4=compute_rectangle_sum(image,i+h/2,j+w/2,h/2,w/2,w_size,M,N);

        f=((S1-S2-S3+S4)*a)/(h*w);
    }

    return f;
}


int detect_face(int x,int y,int w_size,float* image,float std,int M,int N,
                 int* levels,float* shifter,int* feature_index,float* err,
                 float* threshold,int* toggle,Feature* real_features) {
    int k,l,f_idx,r_idx;
    float weight,f_value;
    float label;

    for (l=0;l<MAX_CASCADE_LAYERS;l++)
    {
        label=0;

        for (r_idx=0;r_idx<levels[l];r_idx++)
        {

            f_idx=feature_index[l*levels[l]+r_idx];
            f_value=feature_scaling(real_features,image,x,y,w_size,f_idx,true,M,N)/std;
            weight=((f_value>threshold[l*levels[l]+r_idx])?1:-1)*toggle[l*levels[l]+r_idx]+shifter[l];

            if (err[l*levels[l]+r_idx]==0) {
                if (r_idx==0)
                    return weight>0?1:0;
                return 0;
            }
            label+=weight*log(1/err[l*levels[l]+r_idx]-1);
        }

        if (label<0)
            return 0;
   }

   return 1;
}

void post_processing(int M,int N) {
    int i,j,k,nFriends;
    int *E=(int*)malloc(N*M*sizeof(int));

    sort(face_windows.begin(),face_windows.end(),window_order);

    for (k=0;k<face_windows.size();k++) {
        i=face_windows[k].x;
        j=face_windows[k].y;
        E[i*N+j]=face_windows[k].h;
    }


    int *conn_parts=(int*)malloc(N*M*sizeof(int));
    ConnectedComponents cc(10);
    cc.connected( E, conn_parts, min(N,M), max(N,M), std::equal_to<int>(), true );

    vector<int> mv(conn_parts,conn_parts+N*M);
    sort(mv.begin(),mv.end());
    vector<int> part_count;
    part_count.push_back(1);
    vector<int> part_index;
    part_index.push_back(mv[0]);
    int now_at=mv[0];
    int count=0;

    for (i=1;i<N*M;i++) {
        if (now_at!=mv[i]) {
            now_at=mv[i];
            part_index.push_back(now_at);
            part_count.push_back(1);
            count+=1;
        }
        else {
            part_count[count]+=1;
        }
    }
    now_at+=1;
    vector<bool> flags;
    flags.resize(now_at);
    for (i=0;i<now_at;i++)
        flags[i]=true;
    int *fp=(int*)malloc(N*M*sizeof(int));
    for (i=0;i<M;i++)
        for (j=0;j<N;j++)
            fp[i*N+j]=conn_parts[i*N+j];
    vector<Window> representatives;
    vector<float> ratios;
    for (k=0;k<face_windows.size();k++) {
        int part=fp[face_windows[k].x*N+face_windows[k].y];
        if (flags[part]) {
            for (count=0;part_index[count]!=part;count++)
                nFriends=part_count[count];
            if (nFriends>0.5*face_windows[k].h/(float)SAMPLE_SIZE) {
                representatives.push_back(face_windows[k]);
                ratios.push_back((float)nFriends/face_windows[k].h);
            }
            flags[part]=false;
        }
    }
    delete [] conn_parts;
    int nRepresentatives=representatives.size();
    flags.resize(nRepresentatives);
    for (int r=0;r<nRepresentatives;r++)
        flags[r]=true;
    for (int r=0;r<nRepresentatives;r++)
        for (int br=r+1;br<nRepresentatives;br++) {
            if (flags[br] && is_inside(representatives[r],representatives[br])) {
                if (ratios[r]>ratios[br]) {
                    flags[br]=false;
                }
                else {

                    flags[r]=false;
                    break;
                }
            }
        }
    face_windows.resize(0);
    for (int r=0;r<nRepresentatives;r++)
        if (flags[r])
            face_windows.push_back(representatives[r]);
}

int main(int argc, char* argv[]) {
	if(argc<2)
        {
                cout<<"Not enough input parameters"<<endl;
                return 0;
        }
	cout << argv[1] << endl;
	int i,j,i1,j1,k,l,N,M,nr_colors,ok,f_count=0;
    int f_idx,r_idx,scaled_size;
    float scale,std,variance,variance1,variance2,weight,f_value;

        char type[256];

        int *height=(int*)calloc(1,sizeof(int));
        int *width=(int*)calloc(1,sizeof(int));
        int *w_count=(int*)malloc(sizeof(int));
        int* levels=(int*)calloc(MAX_CASCADE_LAYERS,sizeof(int));
        float* shifter=(float*)calloc(MAX_CASCADE_LAYERS,sizeof(float));

        int* feature_index=(int*)calloc(MAX_CASCADE_LAYERS*MAX_RULES_SIZE,sizeof(int));
        float* error=(float*)calloc(MAX_CASCADE_LAYERS*MAX_RULES_SIZE,sizeof(float));
        float* threshold=(float*)calloc(MAX_CASCADE_LAYERS*MAX_RULES_SIZE,sizeof(float));
        int* toggle=(int*)calloc(MAX_CASCADE_LAYERS*MAX_RULES_SIZE,sizeof(int));

        FILE* image_in=fopen(argv[1],"rb");

        ifstream cascade_file("trained_cascade.txt");

        face_windows.clear();

        for (l=0;l<31;l++) {
            int size;
            float shift;
            //fscanf(cascade_file,"%d %f\n",&size,&shift);
            cascade_file>>size>>shift;
            levels[l]=size;
            shifter[l]=shift;

            for (j=0;j<levels[l];j++) {
                cascade_file>>feature_index[l*levels[l]+j]>>error[l*levels[l]+j]>>threshold[l*levels[l]+j]>>toggle[l*levels[l]+j];
            }
        }
        cascade_file.close();
        //fclose(cascade_file);

        fscanf(image_in,"%s\n%d %d\n%d\n",type,&N,&M,&nr_colors);
        *height=M;
        *width=N;
	
        cout << "height: "<< M <<" ; width: "<< N <<endl;

        Window* windows=(Window*)calloc(100000000,sizeof(Window));

        unsigned char *R = (unsigned char*)calloc(N*M,sizeof(unsigned char));
        unsigned char *G = (unsigned char*)calloc(N*M,sizeof(unsigned char));
        unsigned char *B = (unsigned char*)calloc(N*M,sizeof(unsigned char));

        unsigned char *r = (unsigned char*)calloc(N*M,sizeof(unsigned char));
        unsigned char *g = (unsigned char*)calloc(N*M,sizeof(unsigned char));
        unsigned char *b = (unsigned char*)calloc(N*M,sizeof(unsigned char));
        
        ifstream feature_file("trained_cascade");
        feature_file>>f_count;
        Feature* ftrs=(Feature*)calloc(MAX_FEATURE_SIZE,sizeof(Feature));
        for (i=0;i<MAX_FEATURE_SIZE;i++) {
            feature_file>>ftrs[i].type>>ftrs[i].x>>ftrs[i].y>>ftrs[i].h>>ftrs[i].w;
        }
        feature_file.close();

        float *gray_image=(float*)calloc(N*M,sizeof(float));
        float *gray_image_square=(float*)calloc(N*M,sizeof(float));

        float *integral_image=(float*)malloc(sizeof(float)*N*M);
        float *integral_image_square=(float*)malloc(sizeof(float)*N*M);

        for (i=0;i<M;i++) {
            for (j=0;j<N;j++) {
                fread(&R[i*N+j],sizeof(unsigned char),1,image_in);
                fread(&G[i*N+j],sizeof(unsigned char),1,image_in);
                fread(&B[i*N+j],sizeof(unsigned char),1,image_in);

                r[i*N+j]=R[i*N+j];
                g[i*N+j]=G[i*N+j];
                b[i*N+j]=B[i*N+j];

                gray_image[i*N+j]=(6969.0*R[i*N+j]+23434.0*G[i*N+j]+2365.0*B[i*N+j])/32768.0;
                gray_image_square[i*N+j]=gray_image[i*N+j]*gray_image[i*N+j];
            }
        }

        fclose(image_in);
        compute_integral_image(gray_image,integral_image,M,N);
        compute_integral_image(gray_image_square,integral_image_square,M,N);

         for (int ij=0;ij<=(N-SAMPLE_SIZE+1)*(M-SAMPLE_SIZE+1);ij++) {
            i=ij/(M-SAMPLE_SIZE+1);
            j=ij%(M-SAMPLE_SIZE+1);

            Window wi;
            wi.x=i;
            wi.y=j;
            wi.h=SAMPLE_SIZE;
            wi.w=SAMPLE_SIZE;
            scale=1;
            
            scaled_size=SAMPLE_SIZE;
            while (i+wi.h<=M && j+wi.w<=N) {
                variance1=compute_rectangle_sum(integral_image_square,i,j,wi.h,wi.w,scaled_size,M,N)/((float)wi.h*(float)wi.h);
                variance2=compute_rectangle_sum(integral_image,i,j,wi.h,wi.w,scaled_size,M,N)/((float)wi.h*(float)wi.h);
                variance1=variance1-variance2*variance2;
                std=sqrt(variance1);

                if (!(isnan(std) || std<1)) {
                    std=std/STD_NORM_CONST;

                    if (detect_face(i,j,wi.h,integral_image,std,M,N,levels,shifter,feature_index,error,threshold,toggle,ftrs)==1) {
                       face_windows.push_back(wi);
                    }
                }

                scale=scale*SCALE_MULTIPLIER;
                wi.h=wi.w=closest_integer(SAMPLE_SIZE,scale);
            }
        }


        post_processing(M,N);
        for (int k=0;k<face_windows.size();k++) {
            //if (face_windows[k].h<=36) {
                for (i=face_windows[k].x;i<face_windows[k].x+face_windows[k].h;i++) {
                    for (j=face_windows[k].y;j<face_windows[k].y+face_windows[k].w;j++) {
                        bool paint=false;

                            if( abs(face_windows[k].x + PEN_WIDTH/2 -i) <= PEN_WIDTH/2 && abs(face_windows[k].y + face_windows[k].w/2 - j) <= face_windows[k].w/2 )
                                paint = true;
                            else if(abs(face_windows[k].x + face_windows[k].h - 1 - PEN_WIDTH/2 - i) <= PEN_WIDTH/2 && abs(face_windows[k].y + face_windows[k].w/2 - j) <= face_windows[k].w/2)
                                paint = true;
                            else if(abs(face_windows[k].y + PEN_WIDTH/2 -j) <= PEN_WIDTH/2 && abs(face_windows[k].x + face_windows[k].h/2 - i) <= face_windows[k].h/2)
                                paint = true;
                            else if(abs(face_windows[k].y + face_windows[k].w - 1 - PEN_WIDTH/2 - j) <= PEN_WIDTH/2 && abs(face_windows[k].x + face_windows[k].h/2 - i) <= face_windows[k].h/2)
                                paint = true;
                            if (paint) {
                                r[i*N+j]=0;
                                g[i*N+j]=255;
                                b[i*N+j]=0;
                            }
                    }
                }
            //}
        }
        FILE* image_out_seq=fopen(argv[2],"wb");

        fprintf(image_out_seq,"%s\n","P6");
        fprintf(image_out_seq,"%d %d\n",N,M);
        fprintf(image_out_seq,"%d\n",nr_colors);

        for (i=0;i<N;i++)
            for (j=0;j<M;j++) {
                fwrite(&r[i*M+j],sizeof(unsigned char),1,image_out_seq);
                fwrite(&g[i*M+j],sizeof(unsigned char),1,image_out_seq);
                fwrite(&b[i*M+j],sizeof(unsigned char),1,image_out_seq);
            }
        fclose(image_out_seq);
        return 0;
}
