#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "permute.c"

#define bool int
#define TRUE 1
#define FALSE 0
#define g(h,a) (1 / (1 + exp(-(a)*(h))))   /* The activation function */
#define sqr(x) ((x)*(x))
#define abs(x) ( (x<0) ? (-x) : (x) )
/* #define wrandom() ( (((double) lrand48()) / 1073741823.0) - 1.0 ) */
#define wrandom() ( (((double) rand()) / (RAND_MAX/2.0) ) - 1.0 )

#define HID_UNITS 20



/************************************************************************
// This program implements a multi-layer perceptron consisting of an
// input layer, a number of hidden layers (>=1) and an output layer.
//
// The net is trained according to the algoritm 'BP'.
// The teaching is done incrementally i.e. the weights and
// thresholds are adjusted after presentation of each pattern in the
// training set.
//
// A variable 'net' of the type 'Neural_net' has the following records:
//
// net->no_inputs:                   Integer representing #input
// net->no_outputs:                  Integer representing #output
// net->no_hidden_layers:            Integer representing #hidden layers
// net->no_units[]:                  Array of size of the i'th layer
// net->eta, net->alfa:              Learning parameters
// net->acc:                         Soft/hardness of activation function
// net->input[]:                     Array of the net inputs
// net->layer[u].unit[k].act:        Double representing the activity of
//                                   the k'th unit in the u'th layer.
// net->layer[u].unit[k].weight[j]:  Double representing the weight of the
//                                   link between 'unit k in layer u' AND
//                                   'unit j in layer u-1'.
// net->layer[u].unit[k].delta:      Double representing the units 'delta',
//                                   which is computed by back propagation
// net->layer[u].unit[k].prev_dW[j]: Double representing the last weight
//                                   change of the link between 'unit k in
//                                   layer u' AND 'unit j in layer u-1'.
//
// net->input[0], net->layer[0].act[0], net->layer[1].act[0] e.t.c.
// must ALWAYS equal -1.0. (Is done in the initialisation of the net)
// The reason for this, is that the weight of a given unit A to the 0'th
// unit i the previous layer, simulates the threshold of unit A.
**************************************************************************/

typedef struct pattern
{
  double[] input;        /* Array of the pattern input */
  double[] output;       /* Array of the pattern output */
  int class;            /* The class of the pattern */
} pattern;

typedef struct Data_set
{
  int in,out;           /* Integers representing #input, #output */
  int no;               /* No. of patterns in the data set */
  double error;         /* Total error to train the net for */
  pattern[] Plist;       /* Array of the patterns */
  int no_correct;       /* No.of correct data */
  int no_incorrect;     /* No. of incorrect data */
  double per1,per15,gen;/* The 2 classification percents */
} Data_set;

typedef struct netunit
{
  double act;           /* Activity of the unit */
  double[] weight;       /* Array of the unit links weights */
  double delta;         /* The units 'delta' (for Back Propagation) */
  double[] prev_dW;      /* Array with last weightchanges of the links
			   (for 'Back Propagation with 'momentum') */
  double[] EW;           /* Array with weights differentiated */
  double[] best_weight;  /* Array with best weight obtained */
} netunit;

typedef struct netlayer
{
  netunit[] unit;        /* A layer consists of an array of units */
} netlayer;

typedef struct Neural_net
{
  int no_inputs;        /* #input units */
  int no_outputs;       /* #output units */
  int no_hidden_layers; /* #hidden layers */
  int[] no_units;        /* Array of the size of each layers in the net */
  double eta,alfa;      /* Learning parameters */
  double acc;           /* Soft/hardness of activation function */
  double[] input;        /* Array of the net inputs */
  netlayer[] layer;      /* Array of the nets hidden and output layer */
} Neural_net;



/* Allocate net with the specifications from the start of the pat.file */
/* Format: #input, #output, #hidden_layer, #hid1_units, #hid2_units ... */
void Allocate_net(net,no_hidden_units)
     Neural_net[] net;
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
	  net->layer[l].unit[i].prev_dW =
	    (double *) malloc(sizeof(double)*(prev_size+1));
	  net->layer[l].unit[i].EW =
	    (double *) malloc(sizeof(double)*(prev_size+1));
	  net->layer[l].unit[i].best_weight =
	    (double *) malloc(sizeof(double)*(prev_size+1));
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
	  free(net->layer[l].unit[i].prev_dW);
	  free(net->layer[l].unit[i].EW);
	  free(net->layer[l].unit[i].best_weight);
	}
      free(net->layer[l].unit);
    }
  free(net->layer);
  free(net->no_units);
  free(net->input);
}


/* Initialize weights and threshold to ] -3/sqrt(n); 3/sqrt(n) [     */
/* Also every layers unit 0 is given the activity -1.0               */
void Initialize_net(net)
     Neural_net *net;
{
  int i,j,l,prev_size;
  

  net->input[0]=-1.0;
  for (l=0; l <= net->no_hidden_layers; l++)
    {
      net->layer[l].unit[0].act=-1.0;
      prev_size=((l>0) ? (net->no_units[l-1]) : (net->no_inputs));
      printf("Layer %d:  Initializing weights to +/- %6.3f\n",l,(3/sqrt(((double) prev_size))));
      for (i=1; i <= net->no_units[l]; i++)
	{
	  for (j=0; j <= prev_size; j++)
	    {
	      net->layer[l].unit[i].weight[j]=wrandom();
	      net->layer[l].unit[i].weight[j]*= (3/sqrt(((double) prev_size)));
	      net->layer[l].unit[i].prev_dW[j]=0.0;
	      net->layer[l].unit[i].EW[j]=0.0;
	      net->layer[l].unit[i].best_weight[j]=0.0;
	    }
	}
    }
}

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


/* Compute the activity of all units in the net for the given input */
void Activate_net(net)
     Neural_net *net;
{
  register double sum;
  register int i,j,l;
  
  /* Compute the activity of the 1'st layer using the input layer */
  for (i=1; i <= net->no_units[0]; i++)
    {
      sum=0.0;
      for (j=0; j <= net->no_inputs; j++)
	{
	  sum += net->input[j] * net->layer[0].unit[i].weight[j];
	}
      net->layer[0].unit[i].act = g((sum),(net->acc));
      /*printf("(%7.3f %7.3f) ",sum,g((sum),(net->acc)));*/
    }
  /* Compute the activity of the remaining layers */
  for (l=1; l <= net->no_hidden_layers; l++)
    {
      for (i=1; i <= net->no_units[l]; i++)
	{
	  sum=0.0;
	  for (j=0; j <= net->no_units[l-1]; j++)
	    {
	      sum += net->layer[l-1].unit[j].act *
		net->layer[l].unit[i].weight[j];
	    }
	  net->layer[l].unit[i].act = g((sum),(net->acc));
	  /*printf("(%7.3f,%7.3f) ",sum,net->layer[l].unit[i].act);*/
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
  err=0.0; train->per15=train->per1=train->gen=0.0;
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
      else
	{
	  train->gen+=1.0;
	}
    }
  train->per1 /= train->no_incorrect; train->per1 *= 100.0; 
  train->per15 /= train->no_correct; train->per15 *= 100.0;
  train->gen /= train->no; train->gen *= 100;
 
  /* Calculate generalization on VALIDATION set (can be removed) */
  val->per15=val->per1=val->gen=0.0;
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
      else
	{
	  val->gen+=1.0;
	}
    }
  val->per1 /= val->no_incorrect; val->per1 *= 100.0; 
  val->per15 /= val->no_correct; val->per15 *= 100.0;
  val->gen /= val->no; val->gen *= 100;

  return (0.5*err);
}


/* Compute all units 'delta' by 'error back-propagation'     */
/* It is used that g'(h,a)=a*g(h,a)*(1-g(h,a)) evt. (+0.1)?  */
void Error_back_propagation(net,train,u)
     Neural_net *net;
     Data_set *train;
     int u;
{
  register double sum;
  register i,j,l,out_layer;
  
  /* STEP 4 in the BP algorithm p.120 in the book */
  out_layer=net->no_hidden_layers;
  for (i=1; i <= net->no_outputs; i++)
    {
      net->layer[out_layer].unit[i].delta =
	(train->Plist[u].output[i] - net->layer[out_layer].unit[i].act) *
	  (net->layer[out_layer].unit[i].act * net->acc *
	    (1-net->layer[out_layer].unit[i].act)) + 0.0;
    }
  /* STEP 5 in the BP algorithm p.120 in the book */
  for (l=net->no_hidden_layers-1; l >= 0; l--)
    {
      for (i=1; i <= net->no_units[l]; i++)
	{
	  sum=0.0;
	  for (j=1; j <= net->no_units[l+1]; j++)
	    {
	      sum += net->layer[l+1].unit[j].weight[i] *
		net->layer[l+1].unit[j].delta;
	    }
	  net->layer[l].unit[i].delta = sum * net->acc *
	    (net->layer[l].unit[i].act * (1 - net->layer[l].unit[i].act))
	      + 0.0;
	}
    }

  /* Calculate EW(12) for links between input and first layer */
  for (i=0; i <= net->no_inputs; i++)
    {
      for (j=1; j <= net->no_units[0]; j++)
	{
	  net->layer[0].unit[j].EW[i] = net->input[i] *
	    net->layer[0].unit[j].delta;
	}
    }

  /* Calculate EW's for the rest of the layers */
  for (l=0; l<net->no_hidden_layers; l++)
    {
      for (i=0; i <= net->no_units[l]; i++)
	{
	  for (j=1; j <= net->no_units[l+1]; j++)
	    {
	      net->layer[l+1].unit[j].EW[i] = 
		  net->layer[l+1].unit[j].delta *
		  net->layer[l].unit[i].act;
	    }
	}
      
    }

}


/* Update the weights and threshold using the just calculated 'delta' */
/* This procedure performs STEP 6 in the BP algorithm */
void Update_weights(net)
     Neural_net *net;
{
  register double dW;
  register int i,j,l;
  
  /* Beregn vaegtforandringerne for de "sidste" lag i nettet */
  for (l=net->no_hidden_layers; l>=1; l--)
    {
      for (i=1; i <= net->no_units[l]; i++)
	{
	  for (j=0; j <= net->no_units[l-1]; j++)
 	    {
	      dW = net->eta * net->layer[l].unit[i].EW[j];
	      dW += net->alfa * net->layer[l].unit[i].prev_dW[j];
	      net->layer[l].unit[i].weight[j] += dW;
	      net->layer[l].unit[i].prev_dW[j] = dW;
	    }
	}
    }
  /* Beregn vaegtforandringen for forbindelsen mellem input laget og */
  /* det efterfoelgende lag.                                         */
  for (i=1; i <= net->no_units[0]; i++)
    {
      for (j=0; j <= net->no_inputs; j++)
	{
	  dW = net->eta * net->layer[0].unit[i].EW[j];
	  dW += net->alfa * net->layer[0].unit[i].prev_dW[j];
	  net->layer[0].unit[i].weight[j] += dW;
	  net->layer[0].unit[i].prev_dW[j] = dW;
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
  long tid1,tid2;
  
  time(&tid1);
  data->per1=data->per15=data->gen=0.0;
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
      else
	{
	  data->gen++;
	}
     } 
 
  data->per1 /= data->no_incorrect; data->per1 *= 100.0;
  data->per15 /= data->no_correct; data->per15 *= 100.0;
  data->gen /= data->no; data->gen *= 100;
  time(&tid2);
  printf("Time: %ld   Performance: %d\n",tid2-tid1,data->no/(tid2-tid1));
}


void Print_results(data)
     Data_set *data;
{
  printf("Correctly predicted in class / total in class\n");
  printf("---------------------------------------------\n");
  printf("Incorrect seed classified incorrectly (<1%%):  (#incorrect=%d) %7.3f %%\n",data->no_incorrect,data->per1);
  printf("Correct seed classified incorrectly (<15%%):   (#correct=%d) %7.3f %%\n",data->no_correct,data->per15);
  printf("Generalization:  %7.3f\n",data->gen);
}


/* Train the net until the error is less than the given value */
/* The error is written as a function of #epoch in a file */
void Train_for_error(net,train,val,fname)
     Neural_net *net;
     Data_set *train,*val;
     char fname[30];
{
  Permute_set set;
  int u,epoch;
  double err;
  char fna[2][30];
  FILE *fp,*fopen();
  
  strcpy(fna[0],fname); strcpy(fna[1],fname);
  strcat(fname,".error");
  strcat(fna[0],".val1"); strcat(fna[1],".val15");

  fp=fopen(fname,"w");
  fprintf(fp,"# eta=%7.3f, alfa=%7.3f, error=%7.2f\n",
	        net->eta,net->alfa,train->error);
  fclose(fp);
  fp=fopen(fna[0],"w");
  fprintf(fp,"# eta=%7.3f, alfa=%7.3f, error=%7.2f\n",
	        net->eta,net->alfa,train->error);
  fclose(fp);
  fp=fopen(fna[1],"w");
  fprintf(fp,"# eta=%7.3f, alfa=%7.3f, error=%7.2f\n",
	        net->eta,net->alfa,train->error);
  fclose(fp);

  Allocate_set(&set,train->no);
  err = Error(net,train,val);
  printf("Start error=%7.3f\n",err);
  epoch=0;
  while (err > train->error)
    {
      Init_set(&set,train->no);
      epoch++; printf("Epoch %d.  ",epoch);
      while (Size_of_set(&set) != 0)
	{
	  u = Get_random_element(&set);
	  Present_pattern(net,train,u);
	  Activate_net(net);
	  Error_back_propagation(net,train,u);
	  Update_weights(net);
	}

      err = Error(net,train,val);
      printf("Error=%7.3f\n",err);
      fp=fopen(fname,"a"); fprintf(fp,"%d %7.3f\n",epoch,err); fclose(fp);
      fp=fopen(fna[0],"a"); fprintf(fp,"%d %7.3f\n",epoch,val->per1);
      fclose(fp);
      fp=fopen(fna[1],"a"); fprintf(fp,"%d %7.3f\n",epoch,val->per15); 
      fclose(fp);
    }
  Deallocate_set(&set);
}


/* Train the net for a fixed #epoch, and return the best net */
void Train_for_gen(net,train,val,fname,no_epoch)
     Neural_net *net;
     Data_set *train,*val;
     char fname[30];
     int no_epoch;
{
  Permute_set set;
  int u,epoch,minepoch;
  double err,gen,mingen,gen1,gen15;
  char fna[2][30];
  FILE *fp,*fopen();
  
  strcpy(fna[0],fname); strcpy(fna[1],fname);
  strcat(fname,".error");
  strcat(fna[0],".val1"); strcat(fna[1],".val15");

  fp=fopen(fname,"w");
  fprintf(fp,"# eta=%7.3f, alfa=%7.3f, #epoch=%d\n",
	        net->eta,net->alfa,no_epoch);
  fclose(fp);
  fp=fopen(fna[0],"w");
  fprintf(fp,"# eta=%7.3f, alfa=%7.3f, #epoch=%d\n",
	        net->eta,net->alfa,no_epoch);
  fclose(fp);
  fp=fopen(fna[1],"w");
  fprintf(fp,"# eta=%7.3f, alfa=%7.3f, #epoch=%d\n",
	        net->eta,net->alfa,no_epoch);
  fclose(fp);

  Allocate_set(&set,train->no);
  err = Error(net,train,val); printf("Start error=%7.3f\n",err);
  minepoch=epoch=0; mingen=999.0;
  gen1=gen15=0.0;
  while (epoch < no_epoch)
    {
      Init_set(&set,train->no);
      epoch++; /* printf("Epoch %d.  ",epoch); */
      while (Size_of_set(&set) != 0)
	{
	  u = Get_random_element(&set);
	  Present_pattern(net,train,u);
	  Activate_net(net);
	  Error_back_propagation(net,train,u);
	  Update_weights(net);
	}

      err = Error(net,train,val);
      gen = abs((val->per15 >= 15.0)) * 100.0 + val->per1;
      if ( (gen < mingen) && (gen<100.0) )
	{
	  mingen=gen; gen1=val->per1; gen15=val->per15; minepoch=epoch;
	  Store_weights(net);
	}

      /* printf("Error=%7.3f\n",err); */
      fp=fopen(fname,"a"); fprintf(fp,"%d %7.3f\n",epoch,err); fclose(fp);
      fp=fopen(fna[0],"a"); fprintf(fp,"%d %7.3f\n",epoch,val->per1);
      fclose(fp);
      fp=fopen(fna[1],"a"); fprintf(fp,"%d %7.3f\n",epoch,val->per15); 
      fclose(fp);
    }
  printf("Maximum performance reached after %d epochs.\n",minepoch);
  printf("Incorrect seed classified incorrectly (<1%%): %7.3f%%\n",gen1);
  printf("Correct seed classified incorrectly (<15%%): %7.3f%%\n\n",gen15);

  Deallocate_set(&set);
  Restore_weights(net);
}


void Auto_train(net,train,val,test,no_runs,no_epoch,fname)
     Neural_net *net;
     Data_set *train,*val,*test;
     int no_runs,no_epoch;
     char fname[30];
{
  Permute_set set;
  int u,epoch,minepoch,run,Sepoch,SSQepoch;
  int Stid,Sperf,SSQtid,SSQperf;
  long tid1,tid2,tid3,tid4;
  double err,gen,mingen,gen1,gen15;
  double VSgen1,VSSQgen1,VSgen15,VSSQgen15,Sgen1,SSQgen1,Sgen15,SSQgen15;
  char fna[2][30];
  FILE *fp,*fopen();
  
  strcpy(fna[0],fname);   strcpy(fna[1],fname);
  strcat(fna[0],".valres"); strcat(fna[1],".testres");

  fp=fopen(fna[0],"a");
  fprintf(fp,"# eta=%7.3f, alfa=%7.3f, epoch=%d, hidden=%d\n",
	        net->eta,net->alfa,no_epoch,net->no_units[0]);
  fclose(fp);
  fp=fopen(fna[1],"a");
  fprintf(fp,"# eta=%7.3f, alfa=%7.3f, epoch=%d, hidden=%d\n",
	        net->eta,net->alfa,no_epoch,net->no_units[0]);
  fclose(fp);

  Allocate_set(&set,train->no);

  minepoch=epoch=0; gen1=gen15=0.0;

  Sepoch=Stid=Sperf=SSQepoch=SSQtid=SSQperf=0;
  Sgen1=SSQgen1=Sgen15=SSQgen15=0.0;
  VSgen1=VSSQgen1=VSgen15=VSSQgen15=0.0;
  for (run=0; run < no_runs; run++)
    {
      time(&tid1);
      Initialize_net(net);
      mingen=999.0;
      for (epoch=0; epoch < no_epoch; epoch++)
	{
	  Init_set(&set,train->no);
	  /*printf("Run: %d   Epoch: %d.\n",run,epoch);*/
	  while (Size_of_set(&set) != 0)
	    {
	      u = Get_random_element(&set);
	      Present_pattern(net,train,u);
	      Activate_net(net);
	      Error_back_propagation(net,train,u);
	      Update_weights(net);
	    }
	  
	  err = Error(net,train,val);
	  gen = abs((val->per15 > 15.0)) * 100.0 + val->per1;
	  if ( (gen < mingen) )
	    {
	      mingen=gen; gen1=val->per1; gen15=val->per15; minepoch=epoch;
	      Store_weights(net);
	      printf("Epoch: %d   storing 1 pct %.2f 15 pct %.2f \n",epoch,val->per1,val->per15);
	    }
	  
	}
      time(&tid2);
      Restore_weights(net);
      time(&tid3);
      Test_network(net,test);
      time(&tid4);

      Sepoch+=minepoch; SSQepoch+=minepoch*minepoch;
      VSgen1+=gen1; VSSQgen1+=gen1*gen1;
      VSgen15+=gen15; VSSQgen15+=gen15*gen15;
      Sgen1+=test->per1; SSQgen1+=test->per1*test->per1;
      Sgen15+=test->per15; SSQgen15+=test->per15*test->per15;
      Stid=Stid+(tid2-tid1); SSQtid=SSQtid+(tid2-tid1)*(tid2-tid1);
      Sperf=Sperf+(test->no/(tid4-tid3)); SSQperf=SSQperf+(test->no/(tid4-tid3))*(test->no/(tid4-tid3));
      fp=fopen(fna[0],"a");
      fprintf(fp,"%d %8.4f %8.4f\n",minepoch,gen1,gen15);
      fclose(fp);
      fp=fopen(fna[1],"a");
      fprintf(fp,"%d %8.4f %8.4f %d %d %d\n",minepoch,test->per1,test->per15,tid2-tid1,test->no/(tid4-tid3),(tid4-tid3));
      fclose(fp);
    }
  fp=fopen(fna[0],"a");
  Sepoch/=no_runs; SSQepoch/=no_runs;
  VSgen1/=no_runs; VSgen15/=no_runs; VSSQgen1/=no_runs; VSSQgen15/=no_runs;
  Sgen1/=no_runs; Sgen15/=no_runs; SSQgen1/=no_runs; SSQgen15/=no_runs;
  Stid/=no_runs; SSQtid/=no_runs;
  Sperf/=no_runs; SSQperf/=no_runs;
  fprintf(fp,"Epoch: (%d %7.3f)    1%%: (%7.3f %7.3f)   15%%: (%7.3f %7.3f)\n",
	  Sepoch, sqrt( ((double) (SSQepoch-Sepoch*Sepoch)) ),
	  VSgen1, sqrt((VSSQgen1-VSgen1*VSgen1)),
	  VSgen15, sqrt((VSSQgen15-VSgen15*VSgen15)));
  fclose(fp);
  fp=fopen(fna[1],"a");
  fprintf(fp,"Epoch: (%d %7.3f)    1%%: (%7.3f %7.3f)   15%%: (%7.3f %7.3f)\n",
	  Sepoch, sqrt( ((double) (SSQepoch-Sepoch*Sepoch)) ),
	  Sgen1, sqrt((SSQgen1-Sgen1*Sgen1)),
	  Sgen15, sqrt((SSQgen15-Sgen15*Sgen15)));
  fprintf(fp,"Time: (%d %.2f)    Performance: (%d %.2f)\n",
	  Stid,   sqrt( ((double) (SSQtid-Stid*Stid)) ),
	  Sperf,  sqrt( ((double) (SSQperf-Sperf*Sperf)) ));
  fclose(fp);
	  
  Deallocate_set(&set);
}
     

/* Analyze with use of different parameter values */
void Analyze(net,train,val,test,no_runs,no_epoch,fname)
     Neural_net *net;
     Data_set *train,*val,*test;
     int no_runs,no_epoch;
     char fname[30];
{
  int eta,alfa;

  for (eta=0.1; eta<=0.85; eta+=0.15)
    {
      for (alfa=0.1; alfa<=1.0; alfa+=0.20)
	{
	  Set_eta(net,eta);
	  Set_alfa(net,alfa);
	  Auto_train(net,train,val,test,no_runs,no_epoch,fname);
	}
    }

}


void main(argc,argv)
     int argc;
     char *argv[];
{
  Data_set train,test,val;
  Neural_net net;
  long now;
  char fname[50],fname1[50],fname2[50],fname3[50],fna[2][50];
  FILE *fp,*fopen();

  if ((argc!=3) && (argc!=5) && (argc!=6) && (argc!=7) )
    {
      printf("ANTAL SKJULTE KNUDER=%d\n",HID_UNITS);
      printf("USAGE: BP error <file> <error> <eta> <alfa>\n");
      printf("       BP gen <file> <#epoch> <eta> <alfa>\n");
      printf("       BP test <file>\n");
      printf("       BP auto <file> <#runs> <#epoch> <eta> <alfa>\n");
      printf("       BP analyze <file> <#runs> <#epoch>\n");
    }
  else
    {
      Set_acc(&net,1.0);
      
      time(&now); srand(now); /* srand48(now); */
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
/*	  Show_weights(&net);*/
	  printf("\nNet performance on TEST set\n");
	  Test_network(&net,&test); Print_results(&test);
	}
      else if ( (argc==7) && (strcmp(argv[1],"auto")==0) )
	{
	  strcpy(fna[0],fname);   strcpy(fna[1],fname);
	  strcat(fna[0],".valres"); strcat(fna[1],".testres");
	  fp=fopen(fna[0],"w"); fclose(fp);
	  fp=fopen(fna[1],"w"); fclose(fp);
	  
	  printf("Automatic train for %d runs of %d epochs.\n",atoi(argv[3]),
		 atoi(argv[4]));
	  Read_dataset(&net,&train,strcat(fname,".train"));
	  Read_dataset(&net,&val,strcat(fname1,".val"));
	  Read_dataset(&net,&test,strcat(fname2,".test"));
	  Allocate_net(&net,HID_UNITS);
	  /* Initialize_net(&net); */
	  
	  Set_eta(&net,(double) atof(argv[5]));
	  Set_alfa(&net,(double) atof(argv[6]));
	  
	  Auto_train(&net,&train,&val,&test,atoi(argv[3]),atoi(argv[4]),argv[2]);
	}
      else if ( (argc==5) && (strcmp(argv[1],"analyze")==0) )
	{
	  strcpy(fna[0],fname);   strcpy(fna[1],fname);
	  strcat(fna[0],".valres"); strcat(fna[1],".testres");
	  fp=fopen(fna[0],"w"); fclose(fp);
	  fp=fopen(fna[1],"w"); fclose(fp);
	  
	  printf("Analyze with %d runs of %d epochs.\n",atoi(argv[3]),
	     atoi(argv[4]));
	  Read_dataset(&net,&train,strcat(fname,".train"));
	  Read_dataset(&net,&val,strcat(fname1,".val"));
	  Read_dataset(&net,&test,strcat(fname2,".test"));
	  Allocate_net(&net,HID_UNITS);
	  Initialize_net(&net);
	  
	  Analyze(&net,&train,&val,&test,atoi(argv[3]),atoi(argv[4]),argv[2]);
	}
    }
}




