#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "permute.c"

#define bool int
#define TRUE 1
#define FALSE 0
#define sqr(x) ((x)*(x))
#define abs(x) ( (x<0) ? (-x) : (x) )
/* #define wrandom() ( (((double) lrand48()) / 1073741823.0) - 1.0 ) */
#define wrandom() ( (((double) random()) / RAND_MAX) ) 

#define f(h,a) (1 / (1 + exp(-(a)*(h))))   /* The activation function */

#define N 2            /* Number of eigenvectors and value to compute */
#define M 3            /* #calculation step per eigenvector - and value */
#define F 5            /* Every F'th epoch the eigenvalues are computed */
#define HID_UNITS 15   /* #hidden units in the single hidden layer */


/************************************************************************
// This program implements a multi-layer perceptron consisting of an
// input layer, a number of hidden layers (>=1) and an output layer.
//
// The weights and thresholds are adjusted off-line (i.e. after each epoch)
//
// net->input[0], net->layer[0].act[0], net->layer[1].act[0] e.t.c.
// must ALWAYS equal -1.0. (This is done in the initialisation of the net)
// The reason for this, is that the weight of a given unit A to the 0'th
// unit in the previous layer, simulates the threshold of unit A.
**************************************************************************/

typedef struct pattern
{
  double *input;        /* Array of the pattern input */
  double *output;       /* Array of the pattern output */
  int class;            /* The class of the pattern */
} pattern;

typedef struct Data_set
{
  int in,out;           /* Integers representing #input, #output */
  int no;               /* No. of patterns in the data set */
  double error;         /* Total error to train the net for */
  pattern *Plist;       /* Array of the patterns */
  int no_correct;       /* No.of correct data */
  int no_incorrect;     /* No. of incorrect data */
  double per1,per15;    /* Generalization failures for each class */
} Data_set;


typedef struct list
{
  double e[N];
} list;

typedef struct netunit
{
  double act;           /* Activity of the unit */
  double *weight;       /* Array of the unit links weights */
  double *prev_dW;      /* Array with last weightchanges of the links */
  double *best_weight;  /* Array with best weight obtained */

  double phi;           /* 'phi' used in algorithm */
  double my;            /* 'my' used in algorithm */
  double beta;          /* 'beta' used in algorithm */
  double delta;         /* 'delta' used in algorithm */
  double *Hd;           /* Product component */
  double *d;            /* Vector component d */
  list *ev;             /* Array with element of each of the N eigenvectors */
  double *G;            /* Gradient component */
} netunit;

typedef struct netlayer
{
  netunit *unit;        /* A layer consists of an array of units */
} netlayer;

typedef struct Neural_net
{
  int no_inputs;        /* #input units */
  int no_outputs;       /* #output units */
  int no_hidden_layers; /* #hidden layers */
  int *no_units;        /* Array of the size of each layers in the net */
  double eta,alfa;      /* Learning parameters */
  double acc;           /* Soft/hardness of activation function */
  double *input;        /* Array of the net inputs */
  netlayer *layer;      /* Array of the nets hidden and output layer */
  double egval[N];      /* Array with the computed eigenvalues */
} Neural_net;


/* Set the net parameter 'alfa' (Momentum) */
void Set_alfa(net,alfa)
     Neural_net *net;
     double alfa;
{
  net->alfa=alfa;
}

/* Set the net parameter 'eta' (Learning) */
void Set_eta(net,eta)
     Neural_net *net;
     double eta;
{
  net->eta=eta;
}

/* Set the net parameter 'acc' (Activation function) */
void Set_acc(net,acc)
     Neural_net *net;
     double acc;
{
  net->acc=acc;
}


/* Allocate net with the specifications from the start of the pat.file */
/* Format: #input, #output, #hidden_layer, #hid1_units, #hid2_units ... */
void Allocate_net(net,no_hidden_units)
     Neural_net *net;
     int no_hidden_units;
{
  int l,i,prev_size;

  printf("#Hidden units=%d\n",no_hidden_units);
  net->no_hidden_layers=1; /* --- Only 1 hidden layer --- */

  net->input=(double *) malloc(sizeof(double)*(net->no_inputs+1));
  net->layer=(netlayer *) malloc(sizeof(netlayer)*(net->no_hidden_layers+1));
  net->no_units=(int *) malloc(sizeof(int)*(net->no_hidden_layers+1));
  for (l=0; l < net->no_hidden_layers; l++)
    {
      net->no_units[l]=no_hidden_units;
    }
  net->no_units[net->no_hidden_layers]=net->no_outputs;
  for (l=0; l <= net->no_hidden_layers; l++)
    {
      net->layer[l].unit=(netunit *)
	malloc(sizeof(netunit)*(net->no_units[l]+1));
      for (i=1; i <= net->no_units[l]; i++)
	{
	  prev_size=((l>0) ? (net->no_units[l-1]) : (net->no_inputs));
	  net->layer[l].unit[i].weight =
	    (double *) malloc(sizeof(double)*(prev_size+1));
	  net->layer[l].unit[i].best_weight =
	    (double *) malloc(sizeof(double)*(prev_size+1));
	  net->layer[l].unit[i].prev_dW =
	    (double *) malloc(sizeof(double)*(prev_size+1));
	  net->layer[l].unit[i].G =
	    (double *) malloc(sizeof(double)*(prev_size+1));
	  net->layer[l].unit[i].Hd =
	    (double *) malloc(sizeof(double)*(prev_size+1));
	  net->layer[l].unit[i].d =
	    (double *) malloc(sizeof(double)*(prev_size+1));
	  net->layer[l].unit[i].ev =
	    (list *) malloc(sizeof(list)*(prev_size+1));

	  	  
	}
    }
}


/* Deallocate the memory used by the neural net */
void Deallocate_net(net)
     Neural_net *net;
{
  int l,i;
  
  for (l=0; l<=net->no_hidden_layers; l++)
    {
      for (i=1; i <= net->no_units[l]; i++)
	{
	  free(net->layer[l].unit[i].weight);
	  free(net->layer[l].unit[i].best_weight);
	  free(net->layer[l].unit[i].prev_dW);
	  free(net->layer[l].unit[i].G);
	  free(net->layer[l].unit[i].Hd);
	  free(net->layer[l].unit[i].d);
	  free(net->layer[l].unit[i].ev);
	}
      free(net->layer[l].unit);
    }
  free(net->layer);
  free(net->no_units);
  free(net->input);
}


/* Initialize weights and threshold to ] -3/sqrt(n); 3/sqrt(n) [     */
void Initialize_net(net)
     Neural_net *net;
{
  int i,j,l,prev_size;
  
  net->input[0]=1.0;
  for (l=0; l <= net->no_hidden_layers; l++)
    {
      /* BIAS nodes */
      net->layer[l].unit[0].act=1.0;  
      net->layer[l].unit[0].phi=0.0;

      /* Remaining nodes */
      prev_size=((l>0) ? (net->no_units[l-1]) : (net->no_inputs));
      printf("Layer %d:  Initializing weights to +/- %6.3f\n",
	     l,(3/sqrt(((double) prev_size))));
      for (i=1; i <= net->no_units[l]; i++)
	{
	  for (j=0; j <= prev_size; j++)
	    {
	      net->layer[l].unit[i].weight[j]=wrandom();
	      net->layer[l].unit[i].weight[j]*= (3/sqrt(((double) prev_size)));
	      net->layer[l].unit[i].best_weight[j]=0.0;
	      net->layer[l].unit[i].prev_dW[j]=0.0;
	    }
	}
    }
}


/* Read the training set from file */
void Read_dataset(net,data,filename)
     Neural_net *net;
     Data_set *data;
     char *filename;
{
  FILE *fp,*fopen();
  int u,i,j,temp;
  
  if ( (fp = fopen(filename,"r")) == NULL )
    {
      printf("Could not open file '%s' !!!\n",filename);
      exit(-1);
    }
  else
    {
      fscanf(fp,"%d %d",&net->no_inputs,&net->no_outputs);
      fscanf(fp,"%d",&data->no);
      printf("#Data: %d",data->no);
      printf("  Inputs: %d",net->no_inputs);
      printf("  Outputs: %d\n",net->no_outputs);
      data->in = net->no_inputs;
      data->out = net->no_outputs;
      data->Plist=(pattern *) malloc(sizeof(pattern)*data->no);
      data->no_correct=0; data->no_incorrect=0;
      for (u=0; u <= data->no-1; u++)
	{
	  data->Plist[u].input=(double *) malloc(sizeof(double)*(data->in+1));
	  data->Plist[u].output=(double *) malloc(sizeof(double)*(data->out+1));
	  for (j=1; j <= data->in; j++)
	    {
	      fscanf(fp,"%lf",&data->Plist[u].input[j]);
	    } 
	  fscanf(fp,"%d",&temp);
	  data->Plist[u].class=temp;
	  if (temp==0) data->no_correct++; else data->no_incorrect++;
	  for (i=1; i <= data->out; i++)
	    {
	      data->Plist[u].output[i]=((i==(temp+1)) ? (1.0) : (0.0));
	    } 
	}
      fclose(fp);
    }
}


/* Show the weights of all the links in the neural net */
void Show_weights(net)
     Neural_net *net;
{
  int i,j,l,prev_size;
  
  printf("\nInput til");
  for (l=0; l <= net->no_hidden_layers; l++)
    {
      printf("\nLag %d:",l);
      prev_size=((l>0) ? (net->no_units[l-1]) : (net->no_inputs));
      for (i=1; i <= net->no_units[l]; i++)
	{
	  printf("\n");
	  for (j=0; j <= prev_size; j++)
	    {
	      printf("%7.3f ",net->layer[l].unit[i].weight[j]);
	    }
	}
    }
  printf("\n");
}


/* Save the weights of all the net links */
void Save_weights(net,filename)
     Neural_net *net;
     char *filename;
{
  FILE *fp,*fopen();
  int i,j,l,prev_size;
  
  if ( (fp = fopen(filename,"w")) == NULL )
    {
      printf("Could not save file '%s' !!!\n",filename);
      exit(-1);
    }
  else
    {
      for (l=0; l <= net->no_hidden_layers; l++)
	{
	  prev_size=((l>0) ? (net->no_units[l-1]) : (net->no_inputs));
	  for (i=1; i <= net->no_units[l]; i++)
	    {
	      for (j=0; j <= prev_size; j++)
		{
		  fprintf(fp,"%12.7f ",net->layer[l].unit[i].weight[j]);
		}
	      fprintf(fp,"\n");
	    }
	  fprintf(fp,"\n");
	}
      fclose(fp);
    }
}


/* Load the weights of the net links */
void Load_weights(net,filename)
     Neural_net *net;
     char *filename;
{
  FILE *fp,*fopen();
  int i,j,l,prev_size;
  
  if ( (fp = fopen(filename,"r")) == NULL )
    {
      printf("Could not load file '%s' !!!\n",filename);
      exit(-1);
    }
  else
    {
      for (l=0; l <= net->no_hidden_layers; l++)
	{
	  prev_size=((l>0) ? (net->no_units[l-1]) : (net->no_inputs));
	  for (i=1; i <= net->no_units[l]; i++)
	    {
	      for (j=0; j <= prev_size; j++)
		{
		  fscanf(fp,"%lf",&net->layer[l].unit[i].weight[j]);
		}
	    }
	}
      fclose(fp);
    }
}


/* Store the weights of all the net links */
void Store_weights(net)
     Neural_net *net;
{
  int i,j,l,prev_size;
  
  for (l=0; l <= net->no_hidden_layers; l++)
    {
      prev_size=((l>0) ? (net->no_units[l-1]) : (net->no_inputs));
      for (i=1; i <= net->no_units[l]; i++)
	{
	  for (j=0; j <= prev_size; j++)
	    {
	      net->layer[l].unit[i].best_weight[j]=
		net->layer[l].unit[i].weight[j];
	    }
	}
    }
}

/* Restore the weights of all the net links */
void Restore_weights(net)
     Neural_net *net;
{
  int i,j,l,prev_size;
  
  for (l=0; l <= net->no_hidden_layers; l++)
    {
      prev_size=((l>0) ? (net->no_units[l-1]) : (net->no_inputs));
      for (i=1; i <= net->no_units[l]; i++)
	{
	  for (j=0; j <= prev_size; j++)
	    {
	      net->layer[l].unit[i].weight[j]=
		net->layer[l].unit[i].best_weight[j];
	    }
	}
    }
}





/* Present the u'te training pattern to the Neural net */
void Present_pattern(net,data,u)
     Neural_net *net;
     Data_set *data;
     int u;
{
  register int i;
  
  for (i=1; i <= net->no_inputs; i++)
    {
      net->input[i] = data->Plist[u].input[i];
    }
}


/* Forward propagation (STEP 2) */
void Activate_net(net)
     Neural_net *net;
{
  register double v;
  register int i,j,l;

  /* Compute the 1'st hidden layer using the input layer */
  for (i=1; i <= net->no_units[0]; i++)
    {
      v = 0.0;
      net->layer[0].unit[i].phi = 0.0;
      for (j=0; j <= net->no_inputs; j++)
	{
	  v += net->input[j] * net->layer[0].unit[i].weight[j];
	  net->layer[0].unit[i].phi +=
	    net->layer[0].unit[i].d[j] * net->input[j]; /* Da phi(rs)=0 */
	}
      net->layer[0].unit[i].act = f((v),(net->acc));
    }

  /* Compute the remaining layers */
  for (l=1; l <= net->no_hidden_layers; l++)
    {
      for (i=1; i <= net->no_units[l]; i++)
	{
	  v = 0.0;
	  for (j=0; j <= net->no_units[l-1]; j++)
	    {
	      v += net->layer[l-1].unit[j].act *
		   net->layer[l].unit[i].weight[j];
	    }
	  net->layer[l].unit[i].act = f((v),(net->acc));
	  net->layer[l].unit[i].phi = 0.0;
	  for (j=0; j <= net->no_units[l-1]; j++)
	    {
	      net->layer[l].unit[i].phi +=
		net->layer[l].unit[i].d[j] * net->layer[l-1].unit[j].act +
		net->layer[l].unit[i].weight[j] * ff((v),(net->acc)) *
		net->layer[l-1].unit[j].phi;
	    }
	}
    }
}



/* Compute the summed error of the net on all patterns in the dataset */
double Error(net,train,val)
     Neural_net *net;
     Data_set *train,*val;
{
  register double err,maxo;
  register int i,u,maxi=1;
  
  /* Calculate error on TRAINING set + generalization */
  err=0.0; train->per15=train->per1=0.0;
  for (u=0; u < train->no; u++)
    {
      Present_pattern(net,train,u);
      Activate_net(net);
      maxo=0.0;
      for (i=1; i <= net->no_outputs; i++)
	{
	  err += sqr((train->Plist[u].output[i]) -
		      (net->layer[net->no_hidden_layers].unit[i].act));
	  /* Find maximum output */
	  if (net->layer[net->no_hidden_layers].unit[i].act > maxo)
	    {
	      maxo=net->layer[net->no_hidden_layers].unit[i].act;
	      maxi=i;
	    }
	}
      if (train->Plist[u].output[maxi]!=1.0)
	{
	  if (maxi==1) train->per1+=1.0; else train->per15+=1.0;
	}
    }
  train->per1 /= train->no_incorrect; train->per1 *= 100.0; 
  train->per15 /= train->no_correct; train->per15 *= 100.0;
 
  /* Calculate generalization on VALIDATION set (can be removed) */
  val->per15=val->per1=0.0;
  for (u=0; u < val->no; u++)
    {
      Present_pattern(net,val,u);
      Activate_net(net);
      maxo=0.0;
      for (i=1; i <= net->no_outputs; i++)
	{
	  /* Find maximum output */
	  if (net->layer[net->no_hidden_layers].unit[i].act > maxo)
	    {
	      maxo=net->layer[net->no_hidden_layers].unit[i].act;
	      maxi=i;
	    }

	}
      if (val->Plist[u].output[maxi]!=1.0)
	{
	  if (maxi==1) val->per1+=1.0; else val->per15+=1.0;
	}
    }
  val->per1 /= val->no_incorrect; val->per1 *= 100.0; 
  val->per15 /= val->no_correct; val->per15 *= 100.0;

  return (0.5*err);
}


/* Backward propagation (STEP 3 and 4) */
void Error_back_propagation(net,train,p)
     Neural_net *net;
     Data_set *train;
     int p;
{
  register int i,j,l,v,out_layer;
  double u,phi;
  int prev_size;
  
  /* Output layer (STEP 3) */
  out_layer=net->no_hidden_layers;
  for (i=1; i <= net->no_outputs; i++)
    {
      net->layer[out_layer].unit[i].delta=
	ff(v) * (net->layer[out_layer].unit[i].act-train->Plist[p].output[i]);
      net->layer[out_layer].unit[i].beta=0.0;
      net->layer[out_layer].unit[i].my=
	(ff(v)-
	 fff(v)*(train->Plist[p].output[i]-net->layer[out_layer].unit[i].act))
	* net->layer[out_layer].unit[i].phi;
      for (j=0; j <= net->no_units[out_layer-1]; j++)
	{
/*	  net->layer[out_layer].unit[i].Hd +=
	    (net->layer[out_layer].unit[i].delta * ff(v) *
	     net->layer[out_layer-1].unit[j].phi);
	  net->layer[out_layer].unit[i].Hd +=
	    (net->layer[out_layer].unit[i].my * 
	     net->layer[out_layer-1].unit[j].act);*/
	  net->layer[out_layer].unit[i].G[j] +=
	    net->layer[out_layer].unit[j].delta *
	    net->layer[out_layer-1].unit[j].act;
	}
    }
  
  /* Backward propagation (STEP 4)*/
  for (l=net->no_hidden_layers-1; l>=0; l--)
    {
      /* -------------- Calculate MY, DELTA and BETA ------------------- */
      for (i=1; i <= net->no_units[l]; i++)
	{
	  net->layer[l].unit[i].my=0.0;
	  for (j=1; j <= net->no_units[l+1]; j++)
	    {
	      net->layer[l].unit[i].my += net->layer[l+1].unit[j].weight[i] *
		                          net->layer[l+1].unit[j].my;
	      net->layer[l].unit[i].delta +=net->layer[l+1].unit[j].weight[i] *
		                            net->layer[l+1].unit[j].delta;
	      net->layer[l].unit[i].beta +=
		( ff(v) * net->layer[l+1].unit[j].weight[i]*
                          net->layer[l+1].unit[j].beta ) +
		( net->layer[l+1].unit[j].d[i] * ff(v) +
		  net->layer[l+1].unit[j].weight[i] * fff(v) *
                  net->layer[l].unit[i].phi ) * net->layer[l+1].unit[j].delta;
	    }
	  net->layer[l].unit[i].my *= ff(v);
	  net->layer[l].unit[i].delta *= ff(v);
	}

      /* --------------------- Calculate Hd and G ---------------------- */
      for (i=1; i<=net->no_units[l]; i++)
	{
	  prev_size=((l>0) ? (net->no_units[l-1]) : (net->no_inputs));
	  for (j=0; j <= prev_size; j++)
	    {
	      u=((l>0) ? (net->layer[l-1].unit[j].act) : (net->input[j]) );
	      phi=((l>0) ? (net->layer[l-1].unit[j].phi) : (0.0) );
	      net->layer[l].unit[i].Hd[j] += 
		net->layer[l].unit[i].delta * ff(v) * phi +
		(net->layer[l].unit[i].my + net->layer[l].unit[i].beta) * u;
	      net->layer[l].unit[i].G[j] +=
		net->layer[l].unit[i].delta * u;
	    }
	}

    }  


}


/* Update the weights and threshold (currently also using MOMENTUM) */
void Update_weights(net)
     Neural_net *net;
{
  register double dW;
  register int i,j,l;
  int prev_size;
  
  /* Compute weight changes for the net on basis of G */
  for (l=net->no_hidden_layers; l>=0; l--)
    {
      prev_size=((l>0) ? (net->no_units[l-1]) : (net->no_inputs));
      for (i=1; i <= net->no_units[l]; i++)
	{
	  for (j=0; j <= prev_size; j++)
 	    {
	      dW = net->eta * net->layer[l].unit[i].G[j];
	      dW += net->alfa * net->layer[l].unit[i].prev_dW[j];
	      net->layer[l].unit[i].weight[j] += dW;
	      net->layer[l].unit[i].prev_dW[j] = dW;
	    }
	}
    }
}


/* Evaluate the net on a test set (calculate ability of generalization) */
void Test_network(net,data)
     Neural_net *net;
     Data_set *data;
{
  int i,u,output=1;
  double maxo;
  
  data->per1=data->per15=0.0;
  for (u=0; u < data->no; u++)
    {
      Present_pattern(net,data,u);
      Activate_net(net);
      maxo=0.0;
      for (i=1; i <= net->no_outputs; i++)
	{
	  if ( net->layer[net->no_hidden_layers].unit[i].act > maxo)
	    {
	      maxo=net->layer[net->no_hidden_layers].unit[i].act;
	      output=i;
	    }
	}
      if (data->Plist[u].output[output]!=1.0)
	{
	  if (output==1) data->per1+=1.0; else data->per15+=1.0;
	}
     } 
 
  data->per1 /= data->no_incorrect; data->per1 *= 100.0;
  data->per15 /= data->no_correct; data->per15 *= 100.0;
}


void Print_results(data)
     Data_set *data;
{
  printf("Correctly predicted in class / total in class\n");
  printf("---------------------------------------------\n");
  printf("Incorrect seed classified incorrectly (<1%%):  (#incorrect=%d) %7.3f %%\n",data->no_incorrect,data->per1);
  printf("Correct seed classified incorrectly (<15%%):   (#correct=%d) %7.3f %%\n",data->no_correct,data->per15);
}


/* Perform one iteration in calculation of eigenvector and value */
void calc_eigen(net)
     Neural_net *net;
{

}


/* Initialization before calculation of 'Hd' and 'G' */
void Init_calc(net)
     Neural_net *net;
{
  int i,j,l,prev_size;
  
  for (l=0; l <= net->no_hidden_layers; l++)
    {
      prev_size=((l>0) ? (net->no_units[l-1]) : (net->no_inputs));
      for (i=1; i <= net->no_units[l]; i++)
	{
	  for (j=0; j <= prev_size; j++)
	    {
	      net->layer[l].unit[i].G[j]=0.0;
	      net->layer[l].unit[i].Hd[j]=0.0;
	    }
	}
    }
}


/* Perform one epoch while computing Hd and G */
void do_one_epoch(net,train,set)
     Neural_net *net;
     Data_set *train;
     Permute_set *set;
{
  int p;

  Init_set(set,train->no);
  Init_calc(net);
  while (Size_of_set(set) != 0)
    {
      p = Get_random_element(set);
      Present_pattern(net,train,p);
      Activate_net(net);
      Error_back_propagation(net,train,p);
    }
}


/* Train the net until the error is less than the given value */
void Train_for_error(net,train,val,fname)
     Neural_net *net;
     Data_set *train,*val;
     char fname[30];
{
  Permute_set set;
  int epoch,i,m;
  double err;
  char fna[2][30];
  FILE *fp,*fopen();
  
  strcpy(fna[0],fname); strcpy(fna[1],fname);
  strcat(fname,".error");

  fp=fopen(fname,"w");
  fprintf(fp,"# eta=%7.3f, alfa=%7.3f, error=%7.2f\n",
	        net->eta,net->alfa,train->error);
  fclose(fp);

  Allocate_set(&set,train->no);
  err = Error(net,train,val); printf("Start error=%7.3f\n",err);
  epoch=0;
  while (err > train->error)
    {
      if ((epoch % F)==0)
	{
	  for (i=0; i<N; i++)
	    {
	      /* <<INSERT RANDOM D vector>> */
	      for (m=0; m<M; m++)
		{
		  do_one_epoch(net,train,&set);
		  calc_eigen(net);
		}
	      /* <<INSERT calculated vector into 'e[i]'>> */
	    }
	}
      else
	{
	  do_one_epoch(net,train,&set);
	}
      Update_weights(net);
      epoch++; printf("Epoch %d.  ",epoch);
      

      err = Error(net,train,val); printf("Error=%7.3f\n",err);
      fp=fopen(fname,"a"); fprintf(fp,"%d %7.3f\n",epoch,err); fclose(fp);
    }
  Deallocate_set(&set);
}




void main(argc,argv)
     int argc;
     char *argv[];
{
  Data_set train,test,val;
  Neural_net net;
  long now;
  char fname[50],fname1[50],fname2[50],fname3[50];
  
  Set_acc(&net,1.0);
  
  time(&now);
  /*srand48(now);*/ srand(now); 
  strcpy(fname,argv[2]);
  strcpy(fname1,argv[2]);
  strcpy(fname2,argv[2]);
  strcpy(fname3,argv[2]);

  if ( (argc==6) && (strcmp(argv[1],"error")==0) )
    {
      printf("Train for error.\n");
      Read_dataset(&net,&train,strcat(fname,".train"));
      Read_dataset(&net,&val,strcat(fname1,".val"));
      Read_dataset(&net,&test,strcat(fname2,".test"));
      Allocate_net(&net,HID_UNITS);
      Initialize_net(&net);

      train.error=test.error=atof(argv[3]);
      Set_eta(&net,(double) atof(argv[4]));
      Set_alfa(&net,(double) atof(argv[5]));
      
      Show_weights(&net);
      Train_for_error(&net, &train, &val, fname3);
      Save_weights(&net,strcat(argv[2],".weight"));
      printf("\nNet performance on TRAINING set\n");
      Test_network(&net,&train); Print_results(&train);
      printf("\nNet performance on TEST set\n");
      Test_network(&net,&test); Print_results(&test);
    }
  else if ( (argc==6) && (strcmp(argv[1],"gen")==0) )
    {
      printf("Train for a maximum generalization within %d epochs.\n",
	     atoi(argv[3]));
      Read_dataset(&net,&train,strcat(fname,".train"));
      Read_dataset(&net,&val,strcat(fname1,".val"));
      Read_dataset(&net,&test,strcat(fname2,".test"));
      Allocate_net(&net,HID_UNITS);
      Initialize_net(&net);

      Set_eta(&net,(double) atof(argv[4]));
      Set_alfa(&net,(double) atof(argv[5]));
      
      Show_weights(&net);
      Train_for_gen(&net, &train, &val, fname3, atoi(argv[3]) );
      Save_weights(&net,strcat(argv[2],".weight"));
      printf("\nNet performance on TRAINING set\n");
      Test_network(&net,&train); Print_results(&train);
      printf("\nNet performance on TEST set\n");
      Test_network(&net,&test); Print_results(&test);
    }
  else if ( (argc==3) && (strcmp(argv[1],"test")==0) )
    {
      printf("Test the net with with test data.\n");
      Read_dataset(&net,&test,strcat(fname2,".test"));
      Allocate_net(&net,HID_UNITS);
      Initialize_net(&net);

      Load_weights(&net,strcat(fname3,".weight"));
      Show_weights(&net);
      printf("\nNet performance on TEST set\n");
      Test_network(&net,&test); Print_results(&test);
    }
  else
    {
      printf("ANTAL SKJULTE KNUDER=%d\n",HID_UNITS);
      printf("USAGE: HBP error <file> <error> <eta> <alfa>\n");
      printf("       HBP gen <file> <#epoch> <eta> <alfa>\n");
      printf("       HBP test <file>\n");
    }
}



