use crate::linalg::matrix::Matrix;
use cudarc::driver::CudaSlice;
use cudarc::driver::safe::CudaContext;
use std::error::Error;

impl Matrix {
    pub fn summat(self: &Self) -> Result<f32, Box<dyn Error>> {
        let mut a_host: Vec<f32> = vec![0.0f32; self.n_rows * self.n_cols];
        self.data
            .context()
            .default_stream()
            .memcpy_dtoh(&self.data, &mut a_host)?;
        Ok(a_host.iter().fold(0.0, |sum, x| sum + x))
    }

    pub fn prodmat(self: &Self) -> Result<f32, Box<dyn Error>> {
        let mut a_host: Vec<f32> = vec![0.0f32; self.n_rows * self.n_cols];
        self.data
            .context()
            .default_stream()
            .memcpy_dtoh(&self.data, &mut a_host)?;
        Ok(a_host.iter().fold(1.0, |prod, x| prod * x))
    }

    pub fn rowsummat(self: &Self) -> Result<Self, Box<dyn Error>> {
        let mut a_host: Vec<f32> = vec![0.0f32; self.n_rows * self.n_cols];
        self.data
            .context()
            .default_stream()
            .memcpy_dtoh(&self.data, &mut a_host)?;
        let mut rowsums: Vec<f32> = vec![0.0f32; self.n_rows];
        for i in 0..self.n_rows {
            for j in 0..self.n_cols {
                rowsums[i] += a_host[(i * self.n_cols) + j];
            }
        }
        let rowsums_dev: CudaSlice<f32> =
            self.data.context().default_stream().clone_htod(&rowsums)?;
        Ok(Self::new(rowsums_dev, self.n_rows, 1)?)
    }

    pub fn colsummat(self: &Self) -> Result<Self, Box<dyn Error>> {
        let mut a_host: Vec<f32> = vec![0.0f32; self.n_rows * self.n_cols];
        self.data
            .context()
            .default_stream()
            .memcpy_dtoh(&self.data, &mut a_host)?;
        let mut colsums: Vec<f32> = vec![0.0f32; self.n_cols];
        for j in 0..self.n_cols {
            for i in 0..self.n_rows {
                colsums[j] += a_host[(i * self.n_cols) + j];
            }
        }
        let colsums_dev: CudaSlice<f32> =
            self.data.context().default_stream().clone_htod(&colsums)?;
        Ok(Self::new(colsums_dev, self.n_cols, 1)?)
    }

    pub fn rowprodmat(self: &Self) -> Result<Self, Box<dyn Error>> {
        let mut a_host: Vec<f32> = vec![0.0f32; self.n_rows * self.n_cols];
        self.data
            .context()
            .default_stream()
            .memcpy_dtoh(&self.data, &mut a_host)?;
        let mut rowprods: Vec<f32> = vec![1.0f32; self.n_rows];
        for i in 0..self.n_rows {
            for j in 0..self.n_cols {
                rowprods[i] *= a_host[(i * self.n_cols) + j];
            }
        }
        let rowprods_dev: CudaSlice<f32> =
            self.data.context().default_stream().clone_htod(&rowprods)?;
        Ok(Self::new(rowprods_dev, self.n_rows, 1)?)
    }

    pub fn colprodmat(self: &Self) -> Result<Self, Box<dyn Error>> {
        let mut a_host: Vec<f32> = vec![0.0f32; self.n_rows * self.n_cols];
        self.data
            .context()
            .default_stream()
            .memcpy_dtoh(&self.data, &mut a_host)?;
        let mut colprods: Vec<f32> = vec![1.0f32; self.n_cols];
        for j in 0..self.n_cols {
            for i in 0..self.n_rows {
                colprods[j] *= a_host[(i * self.n_cols) + j];
            }
        }
        let colprods_dev: CudaSlice<f32> =
            self.data.context().default_stream().clone_htod(&colprods)?;
        Ok(Self::new(colprods_dev, self.n_cols, 1)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_fold() -> Result<(), Box<dyn Error>> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let a_host: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a_dev: CudaSlice<f32> = stream.clone_htod(&a_host)?;
        let a_matrix = Matrix::new(a_dev, 2, 3)?;

        let sum = a_matrix.summat()?;
        println!("sum = {}", sum);
        assert_eq!(sum, 21.0);

        let prod = a_matrix.prodmat()?;
        println!("prod = {}", prod);
        assert_eq!(prod, 720.0);

        let rowsums = a_matrix.rowsummat()?;
        println!("rowsums = {}", rowsums);
        let mut rowsums_host: Vec<f32> = vec![0.0f32; a_matrix.n_rows];
        stream.memcpy_dtoh(&rowsums.data, &mut rowsums_host)?;
        println!("rowsums_host = {:?}", rowsums_host);
        assert_eq!(rowsums_host, [1.0 + 2.0 + 3.0, 4.0 + 5.0 + 6.0]);

        let colsums = a_matrix.colsummat()?;
        println!("colsums = {}", colsums);
        let mut colsums_host: Vec<f32> = vec![0.0f32; a_matrix.n_cols];
        stream.memcpy_dtoh(&colsums.data, &mut colsums_host)?;
        println!("colsums_host = {:?}", colsums_host);
        assert_eq!(colsums_host, [1.0 + 4.0, 2.0 + 5.0, 3.0 + 6.0]);

        let rowprods = a_matrix.rowprodmat()?;
        println!("rowprods = {}", rowprods);
        let mut rowprods_host: Vec<f32> = vec![0.0f32; a_matrix.n_rows];
        stream.memcpy_dtoh(&rowprods.data, &mut rowprods_host)?;
        println!("rowprods_host = {:?}", rowprods_host);
        assert_eq!(rowprods_host, [1.0 * 2.0 * 3.0, 4.0 * 5.0 * 6.0]);

        let colprods = a_matrix.colprodmat()?;
        println!("colprods = {}", colprods);
        let mut colprods_host: Vec<f32> = vec![0.0f32; a_matrix.n_cols];
        stream.memcpy_dtoh(&colprods.data, &mut colprods_host)?;
        println!("colprods_host = {:?}", colprods_host);
        assert_eq!(colprods_host, [1.0 * 4.0, 2.0 * 5.0, 3.0 * 6.0]);

        Ok(())
    }
}
