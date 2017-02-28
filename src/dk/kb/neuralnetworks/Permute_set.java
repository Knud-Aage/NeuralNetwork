/*********************************************************************
// Flg. kode implementerer en permuteret maengde af heltal, som man //
// efter initialisering kan udtage et element ad gangen fra.        //
*********************************************************************/
package dk.kb.neuralnetworks;

public class Permute_set
{
  private int no_elements;
  private int[] element;

  /* Initialisering af den permuterede maengde */
  public void Init_set(Permute_set set, int max)
  {
    for (int i=0; i < max; i++)
      set.element[i] = i;
    set.no_elements = max;
  }

  /* Returner stoerrelsen af den permuterede maengde */
  int Size_of_set(Permute_set set)
  {
    return (set.no_elements);
  }

  /* Returner tilfaeldigt element fra den permuterede maengde */
  int Get_random_element(Permute_set set)
  {
    int i,chosen;

    i = (int) Math.floor(Math.random() * set.no_elements);
    chosen = set.element[i];
    set.element[i] = set.element[--set.no_elements];
    return chosen;
  }

}
