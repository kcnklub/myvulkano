use std::sync::Arc;

use vulkano::{
    buffer::BufferContents,
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder,
        CommandBufferExecFuture, PrimaryAutoCommandBuffer, RenderPassBeginInfo,
        SubpassContents::Inline,
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::{Device, DeviceExtensions, Queue},
    image::{view::ImageView, SwapchainImage},
    instance::Instance,
    memory::allocator::StandardMemoryAllocator,
    pipeline::{
        graphics::{vertex_input::Vertex, viewport::Viewport},
        GraphicsPipeline, Pipeline, PipelineBindPoint,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass},
    shader::ShaderModule,
    swapchain::{
        acquire_next_image, AcquireError, PresentFuture, Swapchain, SwapchainAcquireFuture,
        SwapchainCreateInfo, SwapchainCreationError, SwapchainPresentInfo,
    },
    sync::{
        self,
        future::{FenceSignalFuture, JoinFuture, NowFuture},
        FlushError, GpuFuture,
    },
};
use vulkano_win::VkSurfaceBuild;
use winit::{
    event_loop::EventLoop,
    window::{Window, WindowBuilder},
};

use crate::{app::Square, shaders};

use super::{buffers::Buffers, vulkano_builder_utils::*};

pub type Fence = FenceSignalFuture<
    PresentFuture<CommandBufferExecFuture<JoinFuture<Box<dyn GpuFuture>, SwapchainAcquireFuture>>>,
>;

#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct Vertex2d {
    #[format(R32G32_SFLOAT)]
    pub position: [f32; 2],
}

pub struct Allocators {
    pub memory: StandardMemoryAllocator,
    pub command_buffer: StandardCommandBufferAllocator,
    pub descriptor_set: StandardDescriptorSetAllocator,
}

impl Allocators {
    pub fn new(device: Arc<Device>) -> Self {
        Allocators {
            memory: StandardMemoryAllocator::new_default(device.clone()),
            command_buffer: StandardCommandBufferAllocator::new(device.clone(), Default::default()),
            descriptor_set: StandardDescriptorSetAllocator::new(device),
        }
    }
}

pub struct Renderer {
    _instance: Arc<Instance>,
    window: Arc<Window>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    swapchain: Arc<Swapchain>,
    images: Vec<Arc<SwapchainImage>>,
    render_pass: Arc<RenderPass>,
    frame_buffers: Vec<Arc<Framebuffer>>,
    allocators: Allocators,
    buffers: Buffers<Vertex2d, shaders::vertex_shader_for_moving::Data>,
    vertex_shader: Arc<ShaderModule>,
    fragment_shader: Arc<ShaderModule>,
    viewport: Viewport,
    pipeline: Arc<GraphicsPipeline>,
    command_buffers: Vec<Arc<PrimaryAutoCommandBuffer>>,
}

impl Renderer {
    pub fn initialize(event_loop: &EventLoop<()>) -> Self {
        let instance = get_configured_vk_instance();

        let surface = WindowBuilder::new()
            .build_vk_surface(event_loop, instance.clone())
            .unwrap();

        let window = surface
            .object()
            .unwrap()
            .clone()
            .downcast::<Window>()
            .unwrap();

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        let (physical_device, queue_family_index) =
            select_physical_device(&instance, &surface, &device_extensions);

        let (device, queue) = get_vk_device_and_queue(
            physical_device.clone(),
            queue_family_index,
            device_extensions,
        );

        let capabilities = physical_device
            .surface_capabilities(&surface, Default::default())
            .unwrap();

        let (swapchain, images) = build_initial_swapchain(
            window.clone(),
            surface,
            device.clone(),
            capabilities,
            physical_device,
        );

        let render_pass = get_render_pass(device.clone(), &swapchain.clone());
        let frame_buffers = get_framebuffers(&images, render_pass.clone());

        let vertex_shader = shaders::vertex_shader_for_moving::load(device.clone())
            .expect("failed to create shader module");
        let fragment_shader = shaders::fragment_shader_for_moving::load(device.clone())
            .expect("failed to create shader module");

        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: window.inner_size().into(),
            depth_range: 0.0..1.0,
        };

        let pipeline = get_pipeline(
            device.clone(),
            vertex_shader.clone(),
            fragment_shader.clone(),
            render_pass.clone(),
            viewport.clone(),
        );

        let allocators = Allocators::new(device.clone());

        let buffers = Buffers::initialize_device_local::<SquareModel>(
            &allocators,
            pipeline.layout().set_layouts().get(0).unwrap().clone(),
            images.len(),
            queue.clone(),
        );

        let command_buffers = create_simple_command_buffers(
            &allocators,
            queue.clone(),
            pipeline.clone(),
            &frame_buffers,
            &buffers,
        );

        Self {
            _instance: instance,
            window,
            device,
            queue,
            swapchain,
            images,
            render_pass,
            frame_buffers,
            allocators,
            buffers,
            vertex_shader,
            fragment_shader,
            viewport,
            pipeline,
            command_buffers,
        }
    }

    pub fn get_image_count(&self) -> usize {
        self.images.len()
    }

    pub fn handle_window_resize(&mut self) {
        self.recreate_swapchain();
        self.viewport.dimensions = self.window.inner_size().into();

        self.pipeline = get_pipeline(
            self.device.clone(),
            self.vertex_shader.clone(),
            self.fragment_shader.clone(),
            self.render_pass.clone(),
            self.viewport.clone(),
        );

        self.command_buffers = create_simple_command_buffers(
            &self.allocators,
            self.queue.clone(),
            self.pipeline.clone(),
            &self.frame_buffers,
            &self.buffers,
        )
    }

    pub fn recreate_swapchain(&mut self) {
        let (new_swapchain, new_images) = match self.swapchain.recreate(SwapchainCreateInfo {
            image_extent: self.window.inner_size().into(),
            ..self.swapchain.create_info()
        }) {
            Ok(r) => r,
            Err(SwapchainCreationError::ImageUsageNotSupported { .. }) => return,
            Err(e) => panic!("Failed to create new swapchain {:?}", e),
        };

        self.swapchain = new_swapchain;
        self.frame_buffers =
            create_framebuffers_from_swapchain_images(&new_images, self.render_pass.clone());
    }

    pub fn acquire_swapchain_image(
        &self,
    ) -> Result<(u32, bool, SwapchainAcquireFuture), AcquireError> {
        acquire_next_image(self.swapchain.clone(), None)
    }

    pub fn synchronize(&self) -> NowFuture {
        let mut now = sync::now(self.device.clone());
        now.cleanup_finished();

        now
    }

    pub fn flush_next_future(
        &self,
        previous_future: Box<dyn GpuFuture>,
        swapchain_acquire_future: SwapchainAcquireFuture,
        image_i: u32,
    ) -> Result<Fence, FlushError> {
        previous_future
            .join(swapchain_acquire_future)
            .then_execute(
                self.queue.clone(),
                self.command_buffers[image_i as usize].clone(),
            )
            .unwrap()
            .then_swapchain_present(
                self.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(self.swapchain.clone(), image_i),
            )
            .then_signal_fence_and_flush()
    }

    pub fn update_uniform(&mut self, index: u32, square: &Square) {
        let mut uniform_content = self.buffers.uniforms[index as usize]
            .0
            .write()
            .unwrap_or_else(|e| panic!("Failed to write to uniform buffer\n{}", e));

        uniform_content.color = square.color.into();
        uniform_content.position = square.position;
    }
}

fn create_simple_command_buffers<V: BufferContents, U: BufferContents>(
    allocators: &Allocators,
    queue: Arc<Queue>,
    pipeline: Arc<GraphicsPipeline>,
    framebuffers: &[Arc<Framebuffer>],
    buffers: &Buffers<V, U>,
) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
    framebuffers
        .iter()
        .enumerate()
        .map(|(i, framebuffer)| {
            let mut builder = AutoCommandBufferBuilder::primary(
                &allocators.command_buffer,
                queue.queue_family_index(),
                vulkano::command_buffer::CommandBufferUsage::MultipleSubmit,
            )
            .unwrap();

            let index_buffer = buffers.get_index();
            let index_buffer_len = index_buffer.len();

            builder
                .begin_render_pass(
                    RenderPassBeginInfo {
                        clear_values: vec![Some([0.1, 0.1, 0.1, 1.0].into())],
                        ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                    },
                    Inline,
                )
                .unwrap()
                .bind_pipeline_graphics(pipeline.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    pipeline.layout().clone(),
                    0,
                    buffers.get_uniform_descriptor_set(i),
                )
                .bind_vertex_buffers(0, buffers.get_vertex())
                .bind_index_buffer(index_buffer)
                .draw_indexed(index_buffer_len as u32, 1, 0, 0, 0)
                .unwrap()
                .end_render_pass()
                .unwrap();

            Arc::new(builder.build().unwrap())
        })
        .collect()
}

pub fn create_framebuffers_from_swapchain_images(
    images: &[Arc<SwapchainImage>],
    render_pass: Arc<RenderPass>,
) -> Vec<Arc<Framebuffer>> {
    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}

pub trait Model<V: BufferContents, U: BufferContents> {
    fn get_indices() -> Vec<u16>;
    fn get_vertices() -> Vec<V>;
    fn get_initial_uniform_data() -> U;
}

pub struct SquareModel;

type UniformData = shaders::vertex_shader_for_moving::Data;

impl Model<Vertex2d, UniformData> for SquareModel {
    fn get_vertices() -> Vec<Vertex2d> {
        vec![
            Vertex2d {
                position: [-0.25, -0.25],
            },
            Vertex2d {
                position: [0.25, -0.25],
            },
            Vertex2d {
                position: [-0.25, 0.25],
            },
            Vertex2d {
                position: [0.25, 0.25],
            },
        ]
    }

    fn get_indices() -> Vec<u16> {
        vec![0, 1, 2, 1, 2, 3]
    }

    fn get_initial_uniform_data() -> UniformData {
        UniformData {
            color: [0.0, 0.0, 0.0].into(),
            position: [0.0, 0.0],
        }
    }
}
