#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

/* #define crandom() ( ((double) lrand48()) / 2147483646.0 ) */
#define crandom() ( ((double) rand()) / RAND_MAX ) 

/* Generate data for the IN-CIRCLE problem */
void generate(no,filename)
     int no;
     char *filename;
{
  int i,z;
  long int tot[2];
  double x,y;
  FILE *fp,*fopen();

  tot[0]=tot[1]=0;
  fp=fopen(filename,"w");

  /* 2 inputs, and 2 output classes (0,1) */
  fprintf(fp,"2\n2\n");
  fprintf(fp,"%d\n",no);
  for (i=0; i<no; i++)
    {
        x=crandom();
	y=crandom();
	if ((x<0.0) || (x>1.0) || (y<0.0) || (y>1.0) )
	  {
	    printf("(x,y)=(%5.3f,%5.3f)\n",x,y);
	    printf("Point out of range !!!\n"); exit(-1);
	  }
        z=((((x-0.5)*(x-0.5)+(y-0.5)*(y-0.5)) <=0.25) ? (1) : (0));
	tot[z]++;
	fprintf(fp,"%7.3f %7.3f %d\n",x,y,z);
    }
  fprintf(fp,"#in circle=%d, #out of circle=%d\n",tot[0],tot[1]);
  fclose(fp);
  printf("Generated:  #in circle=%d, #out of circle=%d\n",tot[0],tot[1]);
}
     

void main(argc,argv)
     int argc;
     char *argv[];
{
  long now;

  time(&now);
  /* srand48(now); */
  srand(now);

  if (argc==3)
    {
      generate(atoi(argv[1]),argv[2]);
    }
  else
    {
      printf("USAGE: circle <#points> <filename>\n");
    }
}
