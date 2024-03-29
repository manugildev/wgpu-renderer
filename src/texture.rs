use anyhow::*;
use image::GenericImageView;
use std::path::Path;


pub struct Texture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
}

impl Texture {
    pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    pub fn load<P: AsRef<Path>>(device: &wgpu::Device, queue: &wgpu::Queue, path: P, is_normal_map: bool,) -> Result<Self> {
        // Needed to appease the borrow checker
        let path_copy = path.as_ref().to_path_buf();
        let label = path_copy.to_str();

        let img = image::open(path)?;
        Self::from_image(device, queue, &img, label, is_normal_map)
    }

    #[allow(dead_code)]
    pub fn from_bytes(device: &wgpu::Device, queue: &wgpu::Queue, bytes: &[u8], label: &str, is_normal_map: bool) -> Result<Self> {
        let img = image::load_from_memory(bytes)?;
        return Self::from_image(device, queue, &img, Some(label), is_normal_map);
    }

    pub fn from_image(device: &wgpu::Device, queue: &wgpu::Queue, img: &image::DynamicImage, label: Option<&str>, is_normal_map: bool) -> Result<Self> {
        let dimensions = img.dimensions();
        let rgba = img.to_rgba8();

        let size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth: 1,
        };

        let texture_desc = wgpu::TextureDescriptor {
            label,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: if is_normal_map { wgpu::TextureFormat::Rgba8Unorm } else { wgpu::TextureFormat::Rgba8UnormSrgb },
            usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
        };

        let texture = device.create_texture(&texture_desc);

        let texture_view = wgpu::TextureCopyView {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
        };

        let texture_data_layout = wgpu::TextureDataLayout {
            offset: 0,
            bytes_per_row: 4 * size.width,
            rows_per_image: size.height,
        };

        queue.write_texture(texture_view, &rgba, texture_data_layout, size,);

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let result = Texture {
            texture,
            view,
            sampler,
        };

        return Ok(result);
    }

    pub fn create_depth_texture(device: &wgpu::Device, sc_desc: &wgpu::SwapChainDescriptor, label: &str) -> Self {
        let size = wgpu::Extent3d {
            width: sc_desc.width,
            height: sc_desc.height,
            depth: 1,
        };

        let texture_desc = wgpu::TextureDescriptor {
            label: Some(label),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::DEPTH_FORMAT,
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT | wgpu::TextureUsage::SAMPLED,
        };

        let texture = device.create_texture(&texture_desc);
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler_desc = wgpu::SamplerDescriptor {
            label: Some("DepthSamplerDescriptor"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            lod_min_clamp: -100.0,
            lod_max_clamp: 100.0,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        };
        let sampler = device.create_sampler(&sampler_desc);

        return Self {texture, view, sampler};
    }
}
