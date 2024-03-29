use anyhow::*; // INFO: To use .context("Result message")
use glob::glob;
use std::fs::{read_to_string, write};
use std::path::PathBuf;
use std::env;
use fs_extra::copy_items;
use fs_extra::dir::CopyOptions;
use rayon::prelude::*;

struct ShaderData {
    src: String,
    src_path: PathBuf,
    spv_path: PathBuf,
    kind: shaderc::ShaderKind,
}

impl ShaderData {
    pub fn load(src_path: PathBuf) -> Result<Self> {
        let extension = src_path
            .extension()
            .context("File has no extension")?
            .to_str()
            .context("Extension cannot be converted to &str")?;
        let kind = match extension {
            "vert" => shaderc::ShaderKind::Vertex,
            "frag" => shaderc::ShaderKind::Fragment,
            "comp" => shaderc::ShaderKind::Compute,
            _ => bail!("Unsupported shader: {}", src_path.display()),
        };

        let src = read_to_string(src_path.clone())?;
        let spv_path = src_path.with_extension(format!{"{}.spv", extension});

        return Ok(Self{
            src,
            src_path,
            spv_path,
            kind,
        });

    }
}

fn main() -> Result<()> {
    // Collect all shaders recursively within /src/
    let mut shader_paths = Vec::new();
    shader_paths.extend(glob("./shaders/**/*.vert")?);
    shader_paths.extend(glob("./shaders/**/*.frag")?);
    shader_paths.extend(glob("./shaders/**/*.comp")?);

    let shaders = shader_paths
        .into_par_iter()
        .map(|glob_result| ShaderData::load(glob_result?))
        .collect::<Vec<Result<_>>>()
        .into_iter() // Convert into either Some(Vec) or None
        .collect::<Result<Vec<_>>>()?;

    let mut compiler = shaderc::Compiler::new().context("Unable to create shader compiler")?;

    for shader in shaders {
        //  Tell cargo to rerun this script if something in /src/ changes.
        println!("cargo:rerun-if-changed={}", shader.src_path.as_os_str().to_str().unwrap());

        let compiled = compiler.compile_into_spirv(
            &shader.src,
            shader.kind,
            &shader.src_path.to_str().unwrap(),
            "main",
            None,
        )?;

        write(shader.spv_path, compiled.as_binary_u8())?;
    }

    println!("cargo:rerun-if-changed=res/*");

    let out_dir = env::var("OUT_DIR")?;
    let mut copy_options = CopyOptions::new();
    copy_options.overwrite = true;
    let mut paths_to_copy = Vec::new();
    paths_to_copy.push("res/");
    copy_items(&paths_to_copy, out_dir, &copy_options)?;

    return Ok(());
}
