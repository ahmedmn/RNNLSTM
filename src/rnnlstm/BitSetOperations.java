package rnnlstm;
import java.util.BitSet;

public class BitSetOperations
{
    public BitSetOperations() {}
    
    public  BitSet getBitSet(int value, int dim) {
        BitSet bits = new BitSet();
        int index = 0;
        while (value != 0L) {
        if (value % 2L != 0) {
            bits.set(dim - index -1);
          }
          ++index;
          value = value >>> 1;
        }
        return bits;
    }
    
    
    public  String toString(double[] d) {
        String output= "[";
        for(int i=0;i<d.length-1;i++) {
            output += d[i] + ", ";
        }
        output += d[d.length-1] + "]";
        return output;
    }

    public  String toString(BitSet d, int numDigits) {
        String output= "[";
        for(int i=0;i<numDigits-1;i++) {
            output += (d.get(i)?1.0:0.0) + ", ";
        }
        output += (d.get(numDigits-1)?1.0:0.0) + "]";
        return output;
    }
}
