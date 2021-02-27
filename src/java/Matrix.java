class Matrix {
  private int rows;
  private int cols;
  private double[][] data;
  public Matrix(int rows, int cols) {
    this.rows = rows;
    this.cols = cols;
    this.data = new double[rows][cols];

    for (int i = 0; i < this.rows; i++) {
      for (int j = 0; j < this.cols; j++) {
        this.data[i][j] = 0;
      }
    }
  }
  public static Matrix fromArray(double[] arr) {
    Matrix m = new Matrix(arr.length, 1);
    for (int i = 0; i < arr.length; i++) {
      m.data[i][0] = arr[i];
    }
    return m;
  }

  public static Matrix subtract(Matrix a, Matrix b) {
    Matrix result = new Matrix(a.rows, a.cols);
    for (int i = 0; i < result.rows; i++) {
      for (int j = 0; j < result.cols; j++) {
        result.data[i][j] = a.data[i][j] - b.data[i][j];
      }
    }
    return result;
  }

  public double[][] toArray() {
    double[][] arr = new double[this.rows][this.cols];
    for (int i = 0; i < this.rows; i++) {
      for (int j = 0; j < this.cols; j++) {
        arr[i][j] = this.data[i][j];
      }
    }
    return arr;
  }

  public void randomize() {
    for (int i = 0; i < this.rows; i++) {
      for (int j = 0; j < this.cols; j++) {
        this.data[i][j] = Math.random() * 2 - 1;
      }
    }
  }

  public static Matrix transpose(Matrix matrix) {
    Matrix result = new Matrix(matrix.cols, matrix.rows);
    for (int i = 0; i < matrix.rows; i++) {
      for (int j = 0; j < matrix.cols; j++) {
        result.data[j][i] = matrix.data[i][j];
      }
    }
    return result;
  }
  public static Matrix multiply(Matrix a, Matrix b) {
    if (a.cols != b.rows) {
      System.out.println("Columns must match rows");
      return new Matrix(0,0);
    }
    Matrix result = new Matrix(a.rows, b.cols);

    for (int i = 0; i < result.rows; i++) {
      for (int j = 0; j < result.cols; j++) {
        double sum = 0;
        for (int k = 0; k < a.cols; k++) {
          sum += a.data[i][k] * b.data[k][j];
        }
        result.data[i][j] = sum;
      }
    }
    return result;
  }
  public void multiply(double n) {
    for (int i = 0; i < this.rows; i++) {
      for (int j = 0; j < this.cols; j++) {
        this.data[i][j] *= n;
      }
    }
  }
  public void multiply(Matrix n){
    for (int i = 0; i < this.rows; i++) {
      for (int j = 0; j < this.cols; j++) {
        this.data[i][j] *= n.data[i][j];
      }
    }
  }

  public void mapsigmoid() {
    for (int i = 0; i < this.rows; i++) {
      for (int j = 0; j < this.cols; j++) {
        double val = this.data[i][j];
        this.data[i][j] = 1/(1+Math.exp(-val));
      }
    }
  }
  public void mapdsigmoid() {
    for (int i = 0; i < this.rows; i++) {
      for (int j = 0; j < this.cols; j++) {
        double val = this.data[i][j];
        this.data[i][j] = val * (1-val);
      }
    }
  }
  public static Matrix mapdsigmoid(Matrix matrix) {
    Matrix result = new Matrix(matrix.rows, matrix.cols);
    for (int i = 0; i < matrix.rows; i++) {
      for (int j = 0; j < matrix.cols; j++) {
        double val = matrix.data[i][j];
        result.data[i][j] = val * (1-val);
      }
    }
    return result;
  }
  public void add(double n) {
    
      for (int i = 0; i < this.rows; i++) {
        for (int j = 0; j < this.cols; j++) {
          this.data[i][j] += n;
        }
      }
  }
  public void add(Matrix n){
    for (int i = 0; i < this.rows; i++) {
      for (int j = 0; j < this.cols; j++) {
        this.data[i][j] += n.data[i][j];
      }
    }
  }
  public void print(){
    System.out.println("{\n\tRows: " + this.rows + "\n\tCols: " + this.cols);
    System.out.println("\t[");
    for(int i = 0; i < this.rows; i++){
      System.out.println("\t\t" + java.util.Arrays.toString(this.data[i]));
    }
    System.out.println("\t]\n}");
  }

}

