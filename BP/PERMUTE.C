#include <stdio.h>
  
/*********************************************************************
// Flg. kode implementerer en permuteret maengde af heltal, som man //
// efter initialisering kan udtage et element ad gangen fra.        //
*********************************************************************/

public class Permute_set
{
  int no_elements;
  int[] element;
} Permute_set;

/* Allokering af hukommelse til maengden */
void Allocate_set(set,max)
     Permute_set[] set;
     int max;
{
  set.element = new int[max]);
}

/* Frigoer hukommelse optaget af maengden
void Deallocate_set(set)
     Permute_set *set;
{
  free(set->element);
}
*/

/* Initialisering af den permuterede maengde */
void Init_set(set,max)
     Permute_set[] set;
     int max;
{
  int i;
  
  for (i=0; i < max; i++)
    {
      set.element[i] = i;
    }
  set.no_elements = max;
}

/* Returner stoerrelsen af den permuterede maengde */
int Size_of_set(set)
     Permute_set *set;
{
  return (set->no_elements);
}

/* Returner tilfaeldigt element fra den permuterede maengde */
int Get_random_element(set)
     Permute_set *set;
{
  int i,chosen;
  
  i = (rand() % set->no_elements);
  chosen = set->element[i];
  set->element[i] = set->element[--set->no_elements];
  return chosen;
}

