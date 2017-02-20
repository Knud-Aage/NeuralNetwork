package dk.kb.neuralnetworks;

public class Main {

    public static void main(String[] args) {
	// write your code here
/*        Data_set train,test,val;
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
/*            strcpy(fname,argv[2]);
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
/*                printf("\nNet performance on TEST set\n");
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
/*
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
*/
    }
}
