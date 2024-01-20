use byteorder::{LittleEndian, ReadBytesExt};
use plotters::prelude::*;
use std::error::Error;
use std::fs::File;
use std::io::{self};
use std::path::Path;

fn read_heatmap_file<P: AsRef<Path>>(filename: P) -> Result<Vec<i32>, io::Error> {
    let mut file = File::open(filename)?;
    let mut data = Vec::new();

    while let Ok(value) = file.read_i32::<LittleEndian>() {
        data.push(value);
    }

    Ok(data)
}

fn draw_heatmap<P: AsRef<Path>>(filename: P,fileout: P, y : usize, x : usize) -> Result<(),Box<dyn Error>>{

    // Read the binary file
    let heatmap_data = read_heatmap_file(filename)?;
    // Visualize the heat map using plotters
    let root = BitMapBackend::new(&fileout, (y as u32, x as u32)).into_drawing_area();
    root.fill(&BLACK)?;

    let max_value = heatmap_data.iter().copied().max().unwrap();
    let min_value= heatmap_data.iter().copied().filter(|x| *x != -1).min().unwrap();
    println!("\nCreating heatmap from binary...");
    for (i, &value) in heatmap_data.iter().enumerate() {
        let x = (i % x) as i32;
        let y = (i / y) as i32;
        if value >= 0{
            let normalized_intensity = (value - min_value) as f64 / (max_value - min_value) as f64;

        // Map the normalized intensity to a gradient of hues in the HSL color space
        let color = RGBColor(
            (normalized_intensity * 255.0) as u8,
            0,
            ((1.0 - normalized_intensity) * 255.0) as u8,
        );
        
        root.draw_pixel((x, y), &color)?;
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let y = 1000 as usize;
    let x = 1000 as usize;
    draw_heatmap("../parallel_simulation.bin", "results/heatmap_parallel.png", y, x)?;
    println!("heatmap saved at 'bin_visualizer/result/heatmap_parallel.png'");
    draw_heatmap("../serial_simulation.bin", "results/heatmap_serial.png", y, x)?;
    println!("heatmap saved at 'bin_visualizer/result/heatmap_serial.png'");
    draw_heatmap("../omp_simulation.bin", "results/heatmap_omp.png", y, x)?;
    Ok(())
}
