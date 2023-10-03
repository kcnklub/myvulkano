use std::sync::Arc;

use vulkano::{
    command_buffer::allocator::StandardCommandBufferAllocator,
    descriptor_set::{self, allocator::StandardDescriptorSetAllocator},
    device::Queue,
    format::Format,
    memory::allocator::{MemoryAllocator, StandardMemoryAllocator},
    render_pass::{RenderPass, Subpass},
};

pub struct TestApp {}

impl TestApp {
    pub fn new(gfx_queue: Arc<Queue>, image_format: vulkano::format::Format) -> Self {
        let mem_allocator = Arc::new(StandardMemoryAllocator::new_default(
            gfx_queue.device().clone(),
        ));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            gfx_queue.device().clone(),
            Default::default(),
        ));
        let descriptor_set = Arc::new(StandardDescriptorSetAllocator::new(
            gfx_queue.device().clone(),
        ));

        Self {}
    }
}

struct RenderPassPlaceOverFrame {
    gfx_queue: Arc<Queue>,
    render_pass: Arc<RenderPass>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
}

impl RenderPassPlaceOverFrame {
    fn new(
        gfx_queue: Arc<Queue>,
        mem_allocator: &impl MemoryAllocator,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        output_format: Format,
    ) -> () {
        let render_pass = vulkano::single_pass_renderpass!(
            gfx_queue.device().clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: output_format,
                    samples: 1,
                },
            },
            pass: {
                color: [color],
                depth_stencil: {},
            },
        )
        .unwrap();
        let subpass = Subpass::from(render_pass.clone(), 0).unwrap();
    }
}
