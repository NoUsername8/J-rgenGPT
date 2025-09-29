using System;
using System.Text;
using System.Linq;
using System.IO;
using System.Diagnostics;
using System.Threading.Tasks;
using System.Reflection;
using System.Text.RegularExpressions;
using System.Collections.Generic;
using System.Threading;
using System.Globalization;
using static Transformer;
using System.Net.Sockets;
public class Transformer {
    double temp;
    int transformerSize;
    int wordSize;
    public Matrix outputMatrix;
    TransformerHead[] heads;
    static Random random = new Random();
    public struct WordEmbed {
        public Vector vector;
        public String meaning;
    }
    public struct DataVessel {
        public MatrixVessel[] transformers;
        public Matrix embeddingMatrix;
        public Dictionary<string, int> dictionary;
        public void Rand(int wordSize) {
            for(int i = 0; i < transformers.Length; i++) {
                transformers[i].UpProjection = new Matrix(wordSize * 4, wordSize);
                transformers[i].DownProjection = new Matrix(wordSize, wordSize * 4);
                transformers[i].Bias1 = new Vector(wordSize * 4);
                transformers[i].Bias2 = new Vector(wordSize);
                transformers[i].Query = new Matrix(wordSize, wordSize);
                transformers[i].Key = new Matrix(wordSize, wordSize);
                transformers[i].Value = new Matrix(wordSize, wordSize);
                transformers[i].UpProjection.Init();
                transformers[i].DownProjection.Init();
                transformers[i].Bias1.Init();
                transformers[i].Bias2.Init();
                transformers[i].Query.Init();
                transformers[i].Key.Init();
                transformers[i].Value.Init();
            }
            embeddingMatrix.Init();
        }
    }
    public struct MatrixVessel {
        public Matrix Query;
        public Matrix Key;
        public Matrix Value;
        public Matrix UpProjection;
        public Matrix DownProjection;
        public Vector Bias1;
        public Vector Bias2;
        public void Combine(MatrixVessel attentionStack, MatrixVessel mlpStack) {
            Query = attentionStack.Query;
            Key = attentionStack.Key;
            Value = attentionStack.Value;
            UpProjection = mlpStack.UpProjection;
            DownProjection = mlpStack.DownProjection;
            Bias1 = mlpStack.Bias1;
            Bias2 = mlpStack.Bias2;
        }
    }
    DataVessel data;
    public class Vector {
        public int size;
        public double[] values;
        public double[] grads;
        public double magnitude {
            get {
                double sum = 0;
                for(int i = 0; i < size; i++) {
                    sum += Math.Pow(values[i], 2);
                }
                return Math.Sqrt(sum);
            }
        }
        public Vector(int size) {
            this.size = size;
            values = new double[size];
            grads = new double[size];
        }
        public void Init() {
            for(int i = 0; i < size; i++) {
                values[i] = (random.NextDouble() * 2 - 1) * 0.1;
            }
            Norm();
        }
        public void Norm() {
            double mag = this.magnitude;
            if(mag < 1e-8) mag = 1e-8;
            for(int i = 0; i < size; i++) {
                values[i] = mag == 0 ? 0 : values[i] /= mag;
            }
        }
        public void Step(double lr) {
            for(int i = 0; i < values.Length; i++) {
                double g = grads[i];
                g = g > 1 ? 1 : g;
                g = g < -1 ? -1 : g;
                values[i] -= g * lr;
            }
            Norm();
            ZeroGrad();
        }
        public void ZeroGrad() {
            for(int i = 0; i < values.Length; i++) {
                grads[i] = 0;
            }
        }
        public Matrix ToMatrix() {
            Matrix vectorMatrix = new Matrix(size, 1);
            for(int i = 0; i < size; i++) {
                vectorMatrix.values[i, 0] = values[i];
            }
            return vectorMatrix;
        }
        public Vector Copy() {
            Vector output = new Vector(size);
            Array.Copy(values, output.values, size);
            return output;
        }
        public override string ToString() {
            string s = "";
            foreach(double d in values) {
                s += d + " ";
            }
            return s;
        }
    }
    public class Matrix {
        public int rows;
        public int cols;
        public double[,] values;
        public double[,] grads;
        public Matrix(int rows, int cols) {
            this.rows = rows;
            this.cols = cols;
            values = new double[rows, cols];
            grads = new double[rows, cols];
        }
        public void Init() {
            for(int x = 0; x < rows; x++) {
                for(int y = 0; y < cols; y++) {
                    values[x, y] = (random.NextDouble() * 2 - 1) * 0.1;
                }
            }
            Norm();
        }
        public void Step(double lr) {
            for(int i = 0; i < rows; i++) {
                for(int j = 0; j < cols; j++) {
                    double g = grads[i, j];
                    g = g > 1 ? 1 : g;
                    g = g < -1 ? -1 : g;
                    values[i, j] -= g * lr;
                }
            }
            Norm();
            ZeroGrad();
        }
        public void ZeroGrad() {
            grads = new double[rows, cols];
        }
        public Vector ToVector() {
            if(cols != 1) throw new Exception("Matrix.ToVector: Matrix formating not correct");
            Vector matrixVector = new Vector(rows);
            for(int x = 0; x < rows; x++) {
                matrixVector.values[x] = values[x, 0];
            }
            return matrixVector;
        }
        public Matrix Copy() {
            Matrix output = new Matrix(rows, cols);
            output.values = (double[,])values.Clone();
            return output;
        }
        public override string ToString() {
            string s = "";
            for(int i = 0; i < rows; i++) {
                for(int j = 0; j < cols; j++) {
                    s += values[i, j] + " ";
                }
                s += "\n ";
            }
            return s;
        }
        public void Norm() {
            double sum = 0;
            foreach(double d in values) {
                sum += Math.Pow(d, 2);
            }
            sum = Math.Sqrt(sum);
            for(int x = 0; x < rows; x++) {
                for(int y = 0; y < cols; y++) {
                    values[x, y] = sum == 0 ? 0 : values[x, y] /= sum;
                }
            }
        }
        public Vector ColVec(int index) {
            Vector v = new Vector(rows);
            for(int x = 0; x < v.size; x++) {
                v.values[x] = values[x, index];
            }
            return v;
        }
        public void SetCol(int index, Vector v) {
            if(v.size != rows || index < 0 || index >= cols) return;
            for(int x = 0; x < rows; x++) {
                values[x, index] = v.Copy().values[x];  
            }
        }
    }
    public static class Utils {
        public static Matrix Softmax(Matrix m, double temp) {
            Vector[] input = Utils.Vectorize(m);
            Vector[] output = new Vector[input.Length];
            for(int j = 0; j < output.Length; j++) {
                Vector v = input[j];
                double max = v.values.Max();
                output[j] = new Vector(v.size);
                double sum = 0;
                for(int i = 0; i < v.size; i++) {
                    sum += Math.Exp((v.values[i] - max) / temp);
                    output[j].values[i] = v.values[i] / (1 + Math.Exp(-1.702 * v.values[i]));
                }
                for(int i = 0; i < v.size; i++) {
                    output[j].values[i] = Math.Exp((v.values[i] - max) / temp) / sum;
                }
            }
            return Utils.Matricise(output);
        }
        public static Vector GELU(Vector v) {
            Vector output = new Vector(v.size);
            for(int i = 0; i < v.size; i++) {
                output.values[i] = v.values[i] / (1 + Math.Exp(-1.702 * v.values[i]));
            }
            return output;
        }
        public static Vector[] GELUDerivative(Vector[] vs) {
            Vector[] output = new Vector[vs.Length];
            for(int j = 0; j < vs.Length; j++) {
                output[j] = new Vector(vs[j].size);
                for(int i = 0; i < vs[j].size; i++) {
                    double exp = Math.Exp(1.702 * vs[j].values[i]);
                    output[j].values[i] = exp * (1.702 * vs[j].values[i] + exp + 1) / Math.Pow(exp + 1, 2);
                }
            }
            return output;
        }
        public static Vector VectorAddition(Vector v1, Vector v2) {
            if(v1.size != v2.size) throw new ArgumentException("Utils.VectorAddition: Sizes not matching");
            Vector output = new Vector(v1.size);
            for(int i = 0; i < v1.size; i++) {
                output.values[i] = v1.values[i] + v2.values[i];
            }
            return output;
        }
        public static Matrix Hadamard(Matrix m1, Matrix m2) {
            if(m1.rows != m2.rows || m1.cols != m2.cols) throw new ArgumentException("Utils.Hadamard: Sizes not matching");
            Matrix output = new Matrix(m1.rows, m1.cols);
            for(int x = 0; x < m1.rows; x++) {
                for(int y = 0; y < m1.cols; y++) {
                    output.values[x, y] = m1.values[x, y] * m2.values[x, y];
                }
            }
            return output;
        }
        public static double CosineSimilarity(Vector v1, Vector v2) {
            if(v1.size != v2.size) throw new ArgumentException("Utils.CosineSimilarity: Sizes not matching");
            double sum = 0;
            for(int i = 0; i < v1.size; i++) {
                sum += v1.values[i] * v2.values[i];
            }
            return (v1.magnitude * v2.magnitude) == 0 ? sum : sum / (v1.magnitude * v2.magnitude);
        }
        public static Matrix MatrixProduct(Matrix m1, Matrix m2) {
            if(m1.cols != m2.rows) throw new ArgumentException("Utils.MatrixProduct: Matrix sizes are incompatible for multiplication");
            Matrix output = new Matrix(m1.rows, m2.cols);
            for(int x = 0; x < output.cols; x++) {
                for(int y = 0; y < output.rows; y++) {
                    double sum = 0;
                    for(int i = 0; i < m1.cols; i++) {
                        sum += m1.values[y, i] * m2.values[i, x];
                    }
                    output.values[y, x] = sum;
                }
            }
            output.Norm();
            return output;
        }
        public static Matrix MatrixSum(Matrix m1, Matrix m2) {
            if(m1.cols != m2.cols || m1.rows != m2.rows) throw new ArgumentException("Utils.MatrixSum: Matrix sizes are incompatible for addition");
            Matrix output = new Matrix(m1.rows, m1.cols);
            for(int x = 0; x < m1.rows; x++) {
                for(int y = 0; y < m1.cols; y++) {
                    output.values[x, y] = m1.values[x, y] + m2.values[x, y];
                }
            }
            return output;
        }
        public static Matrix Transpose(Matrix m) {
            Matrix output = new Matrix(m.cols, m.rows);
            for(int x = 0; x < output.rows; x++) {
                for(int y = 0; y < output.cols; y++) {
                    output.values[x, y] = m.values[y, x];
                }
            }
            return output;
        }
        public static Matrix Diag(Vector v) {
            Matrix output = new Matrix(v.size, v.size);
            for(int x = 0; x < output.rows; x++) {
                for(int y = 0; y < output.cols; y++) {
                    output.values[x, y] = x == y ? v.values[x] : 0;
                }
            }
            return output;
        }
        public static Matrix Matricise(Vector[] vs) {
            Matrix output = new Matrix(vs[0].size, vs.Length);
            for(int x = 0; x < output.rows; x++) {
                for(int y = 0; y < output.cols; y++) {
                    output.values[x, y] = vs[y].values[x];
                }
            }
            return output;
        }
        public static Vector[] Vectorize(Matrix m) {
            Vector[] output = new Vector[m.cols];
            for(int y = 0; y < m.cols; y++) {
                output[y] = new Vector(m.rows);
                for(int x = 0; x < m.rows; x++) {
                    output[y].values[x] = m.values[x, y];
                }
            }
            return output;
        }
        public static Matrix ScalarMatrixProduct(Matrix m, double s) {
            Matrix output = new Matrix(m.rows, m.cols);
            for(int x = 0; x < m.rows; x++) {
                for(int y = 0; y < m.cols; y++) {
                    output.values[x, y] = m.values[x, y] * s;
                }
            }
            return output;
        }
        public static Matrix DSoftmax(Matrix aGrad, Matrix activation) {    
            if(aGrad.rows != activation.rows || aGrad.cols != activation.cols) throw new ArgumentException("Utils.DSoftmax: Matrix sizes are incompatible for derivation");
            Matrix output = new Matrix(aGrad.rows, aGrad.cols);
            for(int x = 0; x < output.cols; x++) {
                Matrix grad = aGrad.ColVec(x).ToMatrix();
                Matrix act = activation.ColVec(x).ToMatrix();
                output.SetCol(x, MatrixProduct(MatrixSum(Diag(act.ToVector()), ScalarMatrixProduct(MatrixProduct(act, Transpose(act)), -1)), grad).ToVector());
            }
            return output;
        }
    }
    public Transformer(int transformerSize, int wordSize, bool custom, double temp, DataVessel input = new DataVessel()) {
        this.transformerSize = transformerSize;
        this.wordSize = wordSize;
        this.temp = temp;
        if(custom) {
            data = input;
        } else {
            data = new DataVessel();
            data.embeddingMatrix = new Matrix(wordSize, 0);
            data.transformers = new MatrixVessel[transformerSize];
            data.dictionary = new Dictionary<string, int>();
            data.Rand(wordSize);
        }
        heads = new TransformerHead[transformerSize];
        for(int i = 0; i < transformerSize; i++) {
            heads[i] = new TransformerHead(wordSize, i, data, temp);
        }
    }
    class TransformerHead {
        MLP mlp;
        Attention att;
        public int index;
        public TransformerHead(int embeddingSize, int index, DataVessel data, double temp) {
            int neuronSize = 4 * embeddingSize;
            this.index = index;
            att = new Attention(embeddingSize, index, data);
            att.temp = temp;
            mlp = new MLP(embeddingSize, neuronSize, index, data);
        }
        public Vector[] Evaluate(Vector[] input) {
           return mlp.Forward(att.Forward(input));
        }
        public Vector[] Adjust(Vector[] gradients, double learningRate) {
            Vector[] output = att.Backward(mlp.Backward(gradients));
            mlp.Update(learningRate);
            att.Update(learningRate);
            return output;
        }
        public MatrixVessel UpdateDials() {
            MatrixVessel vessel = new MatrixVessel();
            vessel.Combine(att.GetDials(), mlp.GetDials());
            return vessel;
        }
    }
    class Attention {
        public double temp = 1;
        Matrix weightsQuery;
        Matrix weightsKey;
        Matrix weightsValue;
        Matrix a_matrix;
        Matrix v_matrix;
        Matrix k_matrix;
        Matrix q_matrix;
        Matrix i_matrix;
        int embeddingSize;
        int index;
        public Attention(int embeddingSize, int index, DataVessel data) {
            this.embeddingSize = embeddingSize;
            this.index = index;
            InsertDials(data);
        }
        public void InsertDials(DataVessel data) {
            weightsQuery = data.transformers[index].Query;
            weightsKey = data.transformers[index].Key;
            weightsValue = data.transformers[index].Value;
        }
        public MatrixVessel GetDials() {
            return new MatrixVessel { Query = this.weightsQuery, Key = this.weightsKey, Value = this.weightsValue};
        }   
        public Vector[] Forward(Vector[] data) {
            Matrix input = Utils.Matricise(data);
            i_matrix = input.Copy();
            Matrix Q = Utils.MatrixProduct(weightsQuery, input);
            q_matrix = Q.Copy();
            Matrix K = Utils.MatrixProduct(weightsKey, input);
            k_matrix = K.Copy();
            Matrix V = Utils.MatrixProduct(weightsValue, input);
            v_matrix = V.Copy();
            Matrix Z = Utils.ScalarMatrixProduct(Utils.MatrixProduct(Utils.Transpose(Q), K), Math.Sqrt(embeddingSize));
            Matrix A = Utils.Softmax(Z, temp);
            a_matrix = A.Copy();
            Matrix O = Utils.MatrixProduct(V, Utils.Transpose(A));
            return Utils.Vectorize(O);
        }
        public Vector[] Backward(Vector[] gradients) {
            Matrix gradient = Utils.Matricise(gradients);
            Matrix vGrad = Utils.MatrixProduct(gradient, a_matrix);
            Matrix aGrad = Utils.MatrixProduct(Utils.Transpose(gradient), v_matrix);
            Matrix zGrad = Utils.DSoftmax(aGrad, a_matrix);
            Matrix qGrad = Utils.MatrixProduct(Utils.ScalarMatrixProduct(k_matrix, 1 / Math.Sqrt(embeddingSize)), Utils.Transpose(zGrad));
            Matrix kGrad = Utils.MatrixProduct(Utils.ScalarMatrixProduct(q_matrix, 1 / Math.Sqrt(embeddingSize)), zGrad);
            weightsQuery.grads = Utils.MatrixProduct(qGrad, Utils.Transpose(i_matrix)).values;
            weightsKey.grads = Utils.MatrixProduct(kGrad, Utils.Transpose(i_matrix)).values;
            weightsValue.grads = Utils.MatrixProduct(vGrad, Utils.Transpose(i_matrix)).values;
            Matrix partialQ = Utils.MatrixProduct(Utils.Transpose(weightsQuery), qGrad);
            Matrix partialK = Utils.MatrixProduct(Utils.Transpose(weightsKey), kGrad);
            Matrix partialV = Utils.MatrixProduct(Utils.Transpose(weightsValue), vGrad);
            Matrix passthrough = Utils.MatrixSum(Utils.MatrixSum(partialQ, partialK), partialV);
            return Utils.Vectorize(passthrough);
        }
        public void Update(double lr) {
            weightsQuery.Step(lr);
            weightsKey.Step(lr);
            weightsValue.Step(lr);
        } 
    }
    class MLP {
        Matrix UpProjection;
        Matrix DownProjection;
        Vector Bias1;
        Vector Bias2;
        int embeddingSize;
        int neuronSize;
        int headIndex;
        Matrix z_matrix;
        Matrix h_matrix;
        Matrix i_matrix;
        public MLP(int embeddingSize, int neuronSize, int headIndex, DataVessel data) {
            this.embeddingSize = embeddingSize;
            this.neuronSize = neuronSize;
            this.headIndex = headIndex;
            InsertDials(data);
        }
        public void InsertDials(DataVessel data) {
            UpProjection = data.transformers[headIndex].UpProjection;
            DownProjection = data.transformers[headIndex].DownProjection;
            Bias1 = data.transformers[headIndex].Bias1;
            Bias2 = data.transformers[headIndex].Bias2;
        }
        public MatrixVessel GetDials() {
            return new MatrixVessel { UpProjection = this.UpProjection, DownProjection = this.DownProjection, Bias1 = this.Bias1, Bias2 = this.Bias2 };
        }
        public Vector[] Forward(Vector[] data) {
            Matrix input = Utils.Matricise(data);
            i_matrix = input.Copy();
            Vector[] ab1 = new Vector[data.Length];
            Vector[] ab2 = new Vector[data.Length];
            for(int i = 0; i < data.Length; i++) {
                ab1[i] = Bias1;
                ab2[i] = Bias2;
            }
            Matrix adjustedBias1 = Utils.Matricise(ab1);
            Matrix adjustedBias2 = Utils.Matricise(ab2);
            Matrix weightedSums = Utils.MatrixSum(Utils.MatrixProduct(UpProjection, input.Copy()), adjustedBias1);
            z_matrix = weightedSums.Copy();
            Vector[] unpackedWeightdSums = Utils.Vectorize(weightedSums);
            for(int i = 0; i < unpackedWeightdSums.Length; i++) {
                unpackedWeightdSums[i] = Utils.GELU(unpackedWeightdSums[i]);
            }
            Matrix activation = Utils.Matricise(unpackedWeightdSums);
            h_matrix = activation.Copy();
            Matrix outputSum = Utils.MatrixProduct(DownProjection, activation);
            Matrix output = Utils.MatrixSum(input, Utils.MatrixSum(outputSum, adjustedBias2));
            return Utils.Vectorize(output);
        }
        public Vector[] Backward(Vector[] gradients) {
            Matrix gradient = Utils.Matricise(gradients);
            Matrix throughput = Utils.MatrixSum(gradient, Utils.MatrixProduct(Utils.Transpose(UpProjection), Utils.Hadamard(Utils.Matricise(Utils.GELUDerivative(Utils.Vectorize(z_matrix))), Utils.MatrixProduct(Utils.Transpose(DownProjection), gradient))));
            DownProjection.grads = Utils.MatrixProduct(gradient, Utils.Transpose(h_matrix)).values;
            Matrix gWRTH = Utils.MatrixProduct(Utils.Transpose(DownProjection), gradient);
            Matrix gWRTZ = Utils.Hadamard(Utils.Matricise(Utils.GELUDerivative(Utils.Vectorize(z_matrix))), gWRTH);
            UpProjection.grads = Utils.MatrixProduct(gWRTZ, Utils.Transpose(i_matrix)).values;
            for(int i = 0; i < Bias1.size; i++) {
                Bias1.grads[i] = 0;
                for(int n = 0; n < gWRTZ.cols; n++) {
                    Bias1.grads[i] += gWRTZ.values[i, n];
                }
            }
            for(int i = 0; i < Bias2.size; i++) {
                Bias2.grads[i] = 0;
                for(int n = 0; n < gradient.cols; n++) {
                    Bias2.grads[i] += gradient.values[i, n];
                }
            }
            return Utils.Vectorize(throughput);
        }
        public void Update(double lr) {
            UpProjection.Step(lr);
            DownProjection.Step(lr);
            Bias1.Step(lr);
            Bias2.Step(lr);
        }
    }
    public string Evaluate(String input) {
        Vector[] payload = Embed(input);
        for(int i = 0; i < transformerSize; i++) {
            payload = heads[i].Evaluate(payload);
        }
        outputMatrix = Utils.MatrixProduct(data.embeddingMatrix, Utils.Matricise(payload));
        return Unembed(payload.Last());
    }
    public void Propagate(Vector[] gradient, double learningRate) {
        Vector[] error = new Vector[gradient.Length];
        for(int i = 0; i < gradient.Length; i++) {
            error[i] = gradient[i].Copy();
        }
        for(int i = 0; i < transformerSize; i++) {
            error = heads[i].Adjust(error, learningRate);
        }
    }
    public Vector[] Embed(String input) {
        List<String> context = Tokenize(input);
        foreach(string word in context) {
            Familiarize(word);
        }
        Vector[] output = new Vector[context.Count];
        for(int i = 0; i < context.Count; i++) {
            if(data.dictionary.TryGetValue(context[i], out int index)) {
                Vector embed = GetEmbedding(index);
                output[i] = Utils.VectorAddition(embed, posEncoding(i));
                output[i].Norm();
            }
        }
        return output;
    }
    public Vector posEncoding(int pos) {
        Vector encoding = new Vector(wordSize);
        for(int i = 0; i < wordSize; i++) {
            double angle = pos / Math.Pow(10000, 2 * (i / 2) / wordSize);
            encoding.values[i] = (i % 2 == 0) ? Math.Sin(angle) : Math.Cos(angle);
        }
        return encoding;
    }
    public Vector GetEmbedding(int index) {
        Vector ouput = new Vector(wordSize);
        for(int i = 0; i < wordSize; i++) {
            ouput.values[i] = data.embeddingMatrix.values[index, i];
        }
        return ouput;
    }
    public Matrix ExpandMatrix(Matrix oldMatrix, int newRowCount) {
        Matrix newMatrix = new Matrix(newRowCount, wordSize);
        for(int i = 0; i < oldMatrix.rows; i++) {
            for(int j = 0; j < oldMatrix.cols; j++) {
                newMatrix.values[i, j] = oldMatrix.values[i, j];
            }
        }
        return newMatrix;
    }
    string Unembed(Vector input) {
        double p = 0.9;
        List<Tuple<int, double>> dictionary = new List<Tuple<int, double>>();
        for(int i = 0; i < data.embeddingMatrix.cols; i++) {
            dictionary.Add(new Tuple<int, double>(i, Utils.Softmax(Utils.MatrixProduct(data.embeddingMatrix, input.ToMatrix()), temp).ToVector().values[i]));
        }
        dictionary.Sort((x, y) => y.Item2.CompareTo(x.Item2));
        Vector indices = new Vector(dictionary.Count);
        Vector nucleus = new Vector(dictionary.Count);
        double sum = 0;
        for(int i = 0; i < dictionary.Count; i++) {
            sum += dictionary[i].Item2;
            nucleus.values[i] = sum <= p ? dictionary[i].Item2 : 0;
            indices.values[i] = dictionary[i].Item1;
        }
        nucleus = Utils.Softmax(nucleus.ToMatrix(), temp).ToVector();
        Vector cumulation = new Vector(nucleus.size);
        for(int i  = 0; i < cumulation.size; i++) {
            double cumulativeSum = 0;
            for(int j = 0; j <= i; j++) {
                cumulativeSum += nucleus.values[j];
            }
            cumulation.values[i] = cumulativeSum;
        }
        Random rnd = new Random();
        double u = rnd.NextDouble();
        int index = 0;
        for(int i = 0; i < cumulation.size; i++) {
            if(cumulation.values[i] >= u) {
                index = i;
                break;
            }
        }
        return data.dictionary.FirstOrDefault(x => x.Value == index).Key;
    }
    public void Familiarize(string token) {
        if(data.dictionary.ContainsKey(token)) return;
        int vocabSize = data.dictionary.Count;
        data.dictionary[token] = vocabSize;
        data.embeddingMatrix = ExpandMatrix(data.embeddingMatrix, vocabSize + 1);
        Vector rand = new Vector(wordSize);
        rand.Init();
        for(int i = 0; i < wordSize; i++) {
            data.embeddingMatrix.values[vocabSize, i] = rand.values[i];
        }
    }
    public static List<String> Tokenize(String input) {
        var matches = Regex.Matches(input, @"\w+|[^\w\s]");
        var result = new List<string>();
        foreach(Match match in matches) {
            result.Add(match.Value);
        }
        return result;
    }
    public void UpdateData() {
        for(int i = 0; i < transformerSize; i++) {
            data.transformers[i] = heads[i].UpdateDials();
        }
    }
    public DataVessel Data() {
        for(int i = 0; i < transformerSize; i++) {
            data.transformers[i] = heads[i].UpdateDials();
        }
        return data;
    }
    public int GetEmbeddingSize() {
        return wordSize;
    }
    public void PrintConfig() {
        Console.WriteLine();
        foreach(TransformerHead t in heads) {
            Console.WriteLine("Up: " + t.index);
            Console.WriteLine(t.UpdateDials().UpProjection.ToString());
            Console.WriteLine();
            Console.WriteLine("Down: " + t.index);
            Console.WriteLine(t.UpdateDials().DownProjection.ToString());
            Console.WriteLine();
            Console.WriteLine("B1: " + t.index);
            Console.WriteLine(t.UpdateDials().Bias1.ToString());
            Console.WriteLine();
            Console.WriteLine("B2: " + t.index);
            Console.WriteLine(t.UpdateDials().Bias2.ToString());
            Console.WriteLine();
        }
    }
    public void Reload(DataVessel data) {
        this.data = data;
    }
    public Transformer Copy() {
        Transformer clone = new Transformer(transformerSize, wordSize, true, temp, this.data);
        return clone;
    }
}
public static class Driver {
    public const int _WINDOWSIZE = 32;
    //Input args:
    //Training without existing parameters      -t <TransformerChainLength> <EmbeddingSize> <Iterations> <learning Rate> <temperature>
    //Training with existing parameters         -t <Iterations> <learningRate> <temperature>
    //Conversing                                -c <wordCount> <temperature> <dynamicTraining>
    static void Run(string[] args) {
        if(args[0] == "-t") {
            if(!File.Exists("trainingData.txt")) {
                Console.WriteLine("No training data");
                return;
            }
            if(File.Exists("parameters.txt")) {
                if(args.Length != 4) {
                    Console.WriteLine(" Wrong parameter count!");
                    return;
                }
                Train(true, args);
            } else {
                if(args.Length != 6) {
                    Console.WriteLine(" Wrong parameter count!");
                    return;
                }
                File.Create("parameters.txt").Close();
                Train(false, args);
            }
        } else if(args[0] == "-c") {
            if(args.Length != 4) {
                Console.WriteLine(" Wrong parameter count!");
                return;
            }
            if(Boolean.Parse(args[3])) {
                Console.WriteLine("Please prompt model parameters: <TransformerChainLength> <EmbeddingSize> <learning Rate> <custom>");
                string input = Console.ReadLine();
                if(input == null) {
                    Console.WriteLine("Failed to create model: No parameters specified.");
                    return;
                }
                if(input.Split(' ').Length != 4) {
                    Console.WriteLine("Failed to create model: Invalid parameters.");
                    return;
                }
                string[] configs = input.Split(' ');
                if(Boolean.Parse(configs[3]) && !File.Exists("parameters.txt")) {
                    Console.WriteLine("Failed to create model: Missing parameters");
                    return;
                }
                Transformer transformer;
                if(Boolean.Parse(configs[3])) {
                    string[] preConfigs = File.ReadAllText("parameters.txt").Split(">", StringSplitOptions.RemoveEmptyEntries)[0].Split(';', StringSplitOptions.RemoveEmptyEntries);
                    transformer = new Transformer(Int32.Parse(preConfigs[0]), Int32.Parse(preConfigs[1]), true, Double.Parse(args[2]), ReadDials());
                } else {
                    transformer = new Transformer(Int32.Parse(configs[0]), Int32.Parse(configs[1]), false, Double.Parse(args[2]));
                }
                Converse(Int32.Parse(args[1]), transformer, true, Double.Parse(configs[2]));
            } else {
                if(!File.Exists("parameters.txt")) {
                    Console.WriteLine("Missing dial-file!");
                    return;
                }
                string[] configs = File.ReadAllText("parameters.txt").Split(">", StringSplitOptions.RemoveEmptyEntries)[0].Split(';', StringSplitOptions.RemoveEmptyEntries);
                Transformer transformer = new Transformer(Int32.Parse(configs[0]), Int32.Parse(configs[1]), true, Double.Parse(args[2]), ReadDials());
                Converse(Int32.Parse(args[1]), transformer, false, Double.Parse(configs[2]));
            }
        } else {
            Console.WriteLine("Unknown parameter: " + args[0]);
            return;
        }
    }
    public static string WebRun(string[] args, string input) {
        if(args.Length != 2) {
            return "Error";
        }
        if(!File.Exists("parameters.txt")) {
            return "Missing dial-file!";
        }
        string[] configs = File.ReadAllText("parameters.txt").Split(">", StringSplitOptions.RemoveEmptyEntries)[0].Split(';', StringSplitOptions.RemoveEmptyEntries);
        Transformer transformer = new Transformer(Int32.Parse(configs[0]), Int32.Parse(configs[1]), true, Double.Parse(args[1]), ReadDials());
        return ReturningConversation(Int32.Parse(args[0]), transformer, false, Double.Parse(configs[2]), input);
    }
    static Transformer.DataVessel ReadDials() {
        string[] lines = File.ReadAllText("parameters.txt").Split(">", StringSplitOptions.RemoveEmptyEntries);
        string[] config = lines[0].Split(";", StringSplitOptions.RemoveEmptyEntries);
        int transformerSize = Int32.Parse(config[0]);
        int embeddingSize = Int32.Parse(config[1]);
        int vocabSize = Int32.Parse(config[2]);
        Transformer.DataVessel vessel = new Transformer.DataVessel();
        vessel.transformers = new Transformer.MatrixVessel[transformerSize];
        vessel.embeddingMatrix = new Transformer.Matrix(vocabSize, embeddingSize);
        vessel.dictionary = new Dictionary<string, int>();
        for(int i = 0; i < transformerSize; i++) {
            int tIndex = 1 + (i * 7);
            double[] up = Array.ConvertAll(lines[tIndex].Split(";", StringSplitOptions.RemoveEmptyEntries), s => double.Parse(s, CultureInfo.InvariantCulture));
            double[] down = Array.ConvertAll(lines[tIndex + 1].Split(";", StringSplitOptions.RemoveEmptyEntries), s => double.Parse(s, CultureInfo.InvariantCulture));
            double[] b1 = Array.ConvertAll(lines[tIndex + 2].Split(";", StringSplitOptions.RemoveEmptyEntries), s => double.Parse(s, CultureInfo.InvariantCulture));
            double[] b2 = Array.ConvertAll(lines[tIndex + 3].Split(";", StringSplitOptions.RemoveEmptyEntries), s => double.Parse(s, CultureInfo.InvariantCulture));
            double[] q = Array.ConvertAll(lines[tIndex + 4].Split(";", StringSplitOptions.RemoveEmptyEntries), s => double.Parse(s, CultureInfo.InvariantCulture));
            double[] k = Array.ConvertAll(lines[tIndex + 5].Split(";", StringSplitOptions.RemoveEmptyEntries), s => double.Parse(s, CultureInfo.InvariantCulture));
            double[] v = Array.ConvertAll(lines[tIndex + 6].Split(";", StringSplitOptions.RemoveEmptyEntries), s => double.Parse(s, CultureInfo.InvariantCulture));
            vessel.transformers[i].UpProjection = Utils.Matricise(embeddingSize * 4, embeddingSize, up);
            vessel.transformers[i].DownProjection = Utils.Matricise(embeddingSize, embeddingSize * 4, down);
            vessel.transformers[i].Bias1 = Utils.Vectorize(b1);
            vessel.transformers[i].Bias2 = Utils.Vectorize(b2);
            vessel.transformers[i].Query = Utils.Matricise(embeddingSize, embeddingSize, q);
            vessel.transformers[i].Key = Utils.Matricise(embeddingSize, embeddingSize, k);
            vessel.transformers[i].Value = Utils.Matricise(embeddingSize, embeddingSize, v);
        }
        int embedIndex = 1 + (transformerSize * 7);
        int index = 0;
        for(int i = embedIndex; i < lines.Length; i++) {
            string[] line = lines[i].Split(";", StringSplitOptions.RemoveEmptyEntries);
            vessel.dictionary[line[0]] = index;
            for(int j = 1; j < line.Length; j++) {
                vessel.embeddingMatrix.values[index, j - 1] = double.Parse(line[j]);
            }
            index++;
        }
        return vessel;
    }
    static void WriteDials(Transformer transformer, bool console) {
        if(console) Console.WriteLine("   ---Updating...");
        transformer.UpdateData();
        if(console) Console.WriteLine("   ---Creating load...");
        StringBuilder data = new StringBuilder(">");
        int embeddingSize = transformer.GetEmbeddingSize();
        int transformerAmount = transformer.Data().transformers.Length;
        int vocabSize = transformer.Data().dictionary.Count;
        data.Append(transformerAmount + ";" + embeddingSize + ";" + vocabSize);
        if(console) Console.WriteLine("   ---Populating parameters...");
        for(int i = 0; i < transformerAmount; i++) {
            data.Append(">");
            foreach(double d in transformer.Data().transformers[i].UpProjection.values) {
                data.Append(d.ToString("R", CultureInfo.InvariantCulture) + ";");
            }
            data.Append(">");
            foreach(double d in transformer.Data().transformers[i].DownProjection.values) {
                data.Append(d.ToString("R", CultureInfo.InvariantCulture) + ";");
            }
            data.Append(">");
            foreach(double d in transformer.Data().transformers[i].Bias1.values) {
                data.Append(d.ToString("R", CultureInfo.InvariantCulture) + ";");
            }
            data.Append(">");
            foreach(double d in transformer.Data().transformers[i].Bias2.values) {
                data.Append(d.ToString("R", CultureInfo.InvariantCulture) + ";");
            }
            data.Append(">");
            foreach(double d in transformer.Data().transformers[i].Query.values) {
                data.Append(d.ToString("R", CultureInfo.InvariantCulture) + ";");
            }
            data.Append(">");
            foreach(double d in transformer.Data().transformers[i].Value.values) {
                data.Append(d.ToString("R", CultureInfo.InvariantCulture) + ";");
            }
            data.Append(">");
            foreach(double d in transformer.Data().transformers[i].Key.values) {
                data.Append(d.ToString("R", CultureInfo.InvariantCulture) + ";");
            }
        }
        if(console) Console.WriteLine("   ---Populating dictionary...");
        Dictionary<string, int> dict = transformer.Data().dictionary;
        Transformer.Matrix m = transformer.Data().embeddingMatrix;
        foreach(var kvp in dict.OrderBy(kv => kv.Value)) {
            string token = kvp.Key;
            int index = kvp.Value;
            data.Append(">" + token + ";");
            for(int j = 0; j < embeddingSize; j++) {
                data.Append(m.values[index, j].ToString("R", CultureInfo.InvariantCulture) + ";");
            }
        }
        if(File.Exists("parameters.txt")) File.Delete("parameters.txt");
        if(console) Console.WriteLine("   ---Flash load to file...");
        FileStream fs = File.Create("parameters.txt");
        StreamWriter sw = new StreamWriter(fs);
        sw.Write(data.ToString());
        sw.Close();
        fs.Close();
        if(console) Console.WriteLine("   ---Done!");
    }
    static string ReturningConversation(int wordCount, Transformer transformer, bool dynamicTraining, double lr, string input) {
        bool chat = true;
        bool responding = false;
        int words;
        string context = "";
        while(chat) {
            if(input == "-exit") {
                chat = false;
                break;
            }
            if(input != null) {
                context += "User   " + input + "   " + "AI   ";
                responding = true;
                words = 0;
                while(responding) {
                    String answer = transformer.Evaluate(context);
                    context += answer + " ";
                    words++;
                    if(words >= wordCount * 2 || (words >= wordCount && (answer == "." || answer == "!" || answer == "?"))) {
                        responding = false;
                        context += "   ";
                    }
                }
                return context;
            }
        }
        return " ";
    }
    static void Converse(int wordCount, Transformer transformer, bool dynamicTraining, double lr) {
        bool chat = true;
        bool responding = false;
        int words;
        string context = "";
        while(chat) {
            string input = Console.ReadLine();
            if(input == "-exit") {
                chat = false;
                break;
            }
            if(input != null) {
                context += "User   " + input + "   " + "AI   ";
                if(dynamicTraining) {
                    Thread dynamicTrainer = new Thread(() => DynamicTrain(transformer.Copy(), context, lr));
                    dynamicTrainer.Start();
                }
                responding = true;
                Console.WriteLine();
                words = 0;
                while(responding) {
                    String answer = transformer.Evaluate(context);
                    context += answer + " ";
                    Console.Write(" " + answer);
                    words++;
                    if(words >= wordCount * 2 || (words >= wordCount && (answer == "." || answer == "!" || answer == "?"))) {
                        responding = false;
                        context += "   ";
                    }
                }
                transformer.Reload(ReadDials());
                Console.WriteLine();
                Console.WriteLine();
            }
        }
    }
    static void DynamicTrain(Transformer transformer, string context, double lr) {
        string[] trainingString = Transformer.Tokenize(context).ToArray();
        for(int i = _WINDOWSIZE - 1; i < trainingString.Length - 1; i++) {
            string window = "";
            for(int j = (i - _WINDOWSIZE) + 1; j <= i; j++) {
                window += trainingString[j] + " ";
            }
            transformer.Evaluate(window);
            Matrix output = transformer.outputMatrix;
            for(int x = 0; x < output.cols; x++) {
                for(int y = 0; y < output.rows; y++) {
                    double d = output.values[y, x];
                    if(double.IsNaN(d) || double.IsInfinity(d)) {
                        Console.WriteLine($"!!! UNEXPECZED ANOMALY: WORD {i} OF {(trainingString.Length - 1)}");
                        Console.WriteLine($"ANOMALY: {d}   LOCATION: {x} {y} IN OUTPUT MATRIX!");
                        while(true) {
                            Thread.Sleep(2000);
                            Console.Beep();
                        }
                    }
                }
            }
            Matrix gradient = CalculateLoss(output, trainingString[i + 1], transformer.GetEmbeddingSize(), transformer, false);
            transformer.Propagate(Transformer.Utils.Vectorize(gradient), lr);
            WriteDials(transformer, false);
        }
    }
    static void Train(bool file, string[] args) {
        Stopwatch trainingsWatch = new Stopwatch();
        trainingsWatch.Start();
        Transformer trafo;
        int it;
        int embedding;
        double lr;
        if(file) {
            it = Int32.Parse(args[1]);
            Console.WriteLine("Creating model from existing file");
            string[] configs = File.ReadAllText("parameters.txt").Split(">", StringSplitOptions.RemoveEmptyEntries)[0].Split(';', StringSplitOptions.RemoveEmptyEntries);
            trafo = new Transformer(Int32.Parse(configs[0]), Int32.Parse(configs[1]), true, Double.Parse(args[3]), ReadDials());
            embedding = Int32.Parse(configs[1]);
            lr = Double.Parse(args[2]);
        } else {
            Console.WriteLine("Creating model with size " + Int32.Parse(args[1]) + " " + Int32.Parse(args[2]));
            trafo = new Transformer(Int32.Parse(args[1]), Int32.Parse(args[2]), false, Double.Parse(args[5]));
            it = Int32.Parse(args[3]);
            embedding = Int32.Parse(args[2]);
            lr = Double.Parse(args[4]);
        }
        string[] trainingString = Transformer.Tokenize(File.ReadAllText("trainingData.txt")).ToArray();
        List<String> tList = trainingString.ToList();
        tList.Add("USER");
        tList.Add("AI");
        tList.Add(":");
        trainingString = tList.ToArray();
        Console.WriteLine("Embedding Vocabulary...");
        for(int i = 0; i < trainingString.Length; i++) {
            trafo.Familiarize(trainingString[i]);
        }
        WriteDials(trafo, true);
        for(int iteration = 0; iteration < it; iteration++) {
            Stopwatch iterationWatch = new Stopwatch();
            iterationWatch.Start();
            Console.WriteLine("Iteration " + (iteration + 1) + " of " + it);
            Matrix gradient = new Matrix(embedding, _WINDOWSIZE);
            int index = 0;
            Console.WriteLine("--Evaluating...");
            Parallel.For(_WINDOWSIZE - 1, trainingString.Length - 1, i => {
                int progress = (int) (100.0 * ((double)index / (double)(trainingString.Length - 1 - _WINDOWSIZE)));
                Console.WriteLine($"  --{index} of {trainingString.Length - 1 - _WINDOWSIZE} in Iteration {iteration}  --> {progress}% ");
                string window = "";
                for(int j = (i - _WINDOWSIZE) + 1; j <= i; j++) {
                    window += trainingString[j] + " ";
                }
                trafo.Evaluate(window);
                Matrix trainingOutput = trafo.outputMatrix;
                for(int x = 0; x < trainingOutput.cols; x++) {
                    for(int y = 0; y < trainingOutput.rows; y++) {
                        double d = trainingOutput.values[y, x];
                        if(double.IsNaN(d) || double.IsInfinity(d)) {
                            Console.WriteLine($"!!! UNEXPECZED ANOMALY: WORD {i} OF {(trainingString.Length - 1)} IN {iteration + 1}");
                            Console.WriteLine($"ANOMALY: {d}   LOCATION: {x} {y} IN OUTPUT MATRIX!");
                            while(true) {
                                Thread.Sleep(2000);
                                Console.Beep();
                            }
                        }
                    }
                }
                gradient = Transformer.Utils.MatrixSum(gradient, CalculateLoss(trainingOutput, trainingString[i + 1], embedding, trafo, /*DEBUG*/false)); ;
                index++;
            });
            gradient = Transformer.Utils.ScalarMatrixProduct(gradient, index);
            Console.WriteLine("--Going back...");
            trafo.Propagate(Transformer.Utils.Vectorize(gradient), lr);
            TimeSpan preWriteTime = iterationWatch.Elapsed;
            string elapsedPreWriteTime = String.Format("{0:00}:{1:00}:{2:00}.{3:00}", preWriteTime.Hours, preWriteTime.Minutes, preWriteTime.Seconds, preWriteTime.Milliseconds / 10);
            Console.WriteLine($"--Iteration pre-write time: {elapsedPreWriteTime}");
            Console.WriteLine("--Writing...");
            WriteDials(trafo, true);
            iterationWatch.Stop();
            TimeSpan iterationTime = iterationWatch.Elapsed;
            string elapsedIterationTime = String.Format("{0:00}:{1:00}:{2:00}.{3:00}", iterationTime.Hours, iterationTime.Minutes, iterationTime.Seconds, iterationTime.Milliseconds / 10);
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine("------------  ITERATION REPORT  ------------");
            Console.WriteLine("Iteration " + (iteration + 1) + " of " + it + " completed.");
            Console.WriteLine("Iteration duration: " + elapsedIterationTime);
            Thread.Sleep(5000);
            Console.WriteLine();
            Console.WriteLine();
        }
        Console.WriteLine("Training completed!");
        trainingsWatch.Stop();
        TimeSpan trainingTime = trainingsWatch.Elapsed;
        string elapsedTrainingsTime = String.Format("{0:00}:{1:00}:{2:00}.{3:00}", trainingTime.Hours, trainingTime.Minutes, trainingTime.Seconds, trainingTime.Milliseconds / 10);
        Console.WriteLine("Training duration: " +  elapsedTrainingsTime);
    }
    static class Utils {
        public static Matrix Matricise(int xSize, int ySize, double[] vals) {
            if(vals.Length != xSize * ySize) throw new ArgumentException("Incorrect file format!");
            Matrix m = new Matrix(xSize, ySize);
            for(int x = 0; x < xSize; x++) {
                for(int y = 0; y < ySize; y++) {
                    m.values[x, y] = vals[x * ySize + y];
                }
            }
            return m;
        }
        public static Vector Vectorize(double[] vals) {
            Vector v = new Vector(vals.Length);
            for(int i = 0; i < vals.Length; i++) {
                v.values[i] = vals[i];
            }
            return v;
        }
    }
    static Matrix CalculateLoss(Matrix networkOutput, string expectation, int eSize, Transformer t, bool console) {
        int targetIndex; 
        if(!t.Data().dictionary.TryGetValue(expectation, out targetIndex)) {
            targetIndex = 0;
        }
        Matrix batchProbs = Transformer.Utils.Softmax(networkOutput, 1);
        Vector probs = batchProbs.ColVec(batchProbs.cols - 1);
        double loss = -Math.Log(probs.values[targetIndex]);
        Vector gradVector = new Vector(probs.size);
        for(int i = 0; i < gradVector.size; i++) {
            gradVector.values[i] = probs.values[i];
        }
        gradVector.values[targetIndex] -= 1.0;
        Matrix grad = new Matrix(eSize, networkOutput.cols);
        for(int i = 0; i < grad.cols; i++) {
            for(int j = 0; j < grad.rows; j++) {
                grad.values[j, i] = gradVector.values[j];
            }
        }
        if(console) Console.WriteLine($"    Cost = {loss}");
        return grad;
    }
    static void PrintVector(Vector v) {
        Console.WriteLine();
        for(int i = 0; i < v.size; i++) {
            Console.Write(v.values[i] + " ");
        }
        Console.WriteLine();
    }
}