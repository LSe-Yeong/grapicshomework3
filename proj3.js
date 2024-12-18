import*as THREE from 'three'
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {vec2, vec3, vec4, mat4, utils} from 'wgpu-matrix';

const loc_position = 3;
const loc_normal = 2;
const loc_inter_stage_normal = 1;

const id_group = 0;
const binding_matrices = 7;

const format_depth_texture = 'depth24plus';


async function main() {
    
    //초기 설정
    const adapter = await navigator.gpu?.requestAdapter();
    const device = await adapter?.requestDevice();
    if(!device) {
        throw Error("WebGPU not supported.");
    }

    const canvas = document.querySelector("#webgpu");
    const context = canvas.getContext("webgpu");
    const preferredFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
        device,
        format: preferredFormat,
    });

    let canvasTexture = context.getCurrentTexture();

    class UI {
        static NONE = 0;
        static ROTATING = 1;
        static TRANSLATING = 2;
        static mouseMove = UI.NONE;
        static camera = {fovy:60, position:vec3.create(0,0,-3), near:0.1, far:100};
        static matrices = {
            P: mat4.perspective(utils.degToRad(UI.camera.fovy), canvasTexture.width/canvasTexture.height, UI.camera.near, UI.camera.far),
            R: mat4.identity(),
        };
        static onmousedown(ev) {
            if(ev.buttons & 1)  { UI.mouseMove = UI.ROTATING; }
            else if(ev.buttons & 4) { UI.mouseMove = UI.TRANSLATING ; }
        };
        static onmouseup(ev) {
            UI.mouseMove = UI.NONE;
        };
        static onmousemove(ev) {
            let offset = [ev.movementX, ev.movementY];
            if (UI.mouseMove == UI.ROTATING) {
                UI.update_VP();
                let axis = unproject_vector([offset[1], offset[0], 0], UI.matrices.VP,
                    [0, 0, canvas.clientWidth, canvas.clientHeight]);
                UI.matrices.R = mat4.rotate(UI.matrices.R, [axis[0], axis[1], axis[2]], utils.degToRad(vec2.lenSq(offset)*0.1)); 
            }
            else if(UI.mouseMove == UI.TRANSLATING) {
                UI.update_VP();
                let by = unproject_vector([offset[0], -offset[1], 0], UI.matrices.VP,
                    [0, 0, canvas.clientWidth, canvas.clientHeight]);
                UI.camera.position = vec3.add(UI.camera.position, vec3.transformMat4(vec3.create(by[0], by[1], by[2]), UI.matrices.R));
            }
        };
        static onwheel(ev) {
            ev.preventDefault();
            UI.camera.position[2] = -Math.max(1, Math.min(-UI.camera.position[2] + ev.deltaY*0.01, 50));
            UI.update_VP();
        };
        static update_VP() {
            UI.matrices.VP = mat4.multiply(mat4.translate(UI.matrices.P, UI.camera.position),UI.matrices.R);
        }
    };

    UI.update_VP();

    canvas.onmousedown = UI.onmousedown;
    canvas.onmouseup = UI.onmouseup;
    canvas.onmousemove = UI.onmousemove;
    window.addEventListener("wheel", UI.onwheel, {passive:false});

    
    const tank = await load_gltf("/resource/tank.glb",device, preferredFormat);
    console.log(tank)

    const depthTexture = device.createTexture({
        size: [canvasTexture.width, canvasTexture.height],
        format: format_depth_texture,
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });


    let M = mat4.identity();

    let MVP;

    function render(time) {
        

        canvasTexture = context.getCurrentTexture();
            
        const encoder = device.createCommandEncoder();
        const renderPass = encoder.beginRenderPass({
            colorAttachments: [{
                view: canvasTexture.createView(),
                loadOp: "clear",
                clearValue: {r:.3, g:.3, b:.3, a:1},
                storeOp: "store",
            }],
            depthStencilAttachment: {
                view: depthTexture.createView(),
                depthClearValue: 1.0,
                depthLoadOp: 'clear',
                depthStoreOp: 'store',
            },
        });

        MVP = mat4.multiply(UI.matrices.VP, M);
        tank.render(renderPass, MVP);
        renderPass.end();
        const commandBuffer = encoder.finish();
        device.queue.submit([commandBuffer]);

        requestAnimationFrame(render);
    }

    requestAnimationFrame(render);

}

async function load_gltf(url, device, preferredFormat) {
    const loader = new GLTFLoader();

    const root = await new Promise((resolve,reject) => {
        loader.load(url,
            (model) => {resolve(model);},
            null,
            (error) => {reject(error);});
    });

        // Mesh 객체 탐색
    const meshes = [];
    root.scene.traverse((child) => {
        if (child.isMesh) {
            meshes.push(child);
        }
    });

    if (meshes.length === 0) {
        throw new Error("No Mesh objects found in the GLTF model.");
    }

    // 첫 번째 메쉬 객체를 선택
    const obj = meshes[0];
    console.log("Mesh object found:", obj);

    if (!obj.geometry || !obj.geometry.attributes) {
        throw new Error("Mesh does not contain geometry attributes.");
    }
        
    const positions = obj.geometry.attributes.position.array;
    const normals = obj.geometry.attributes.normal.array;
    const indices = new Uint32Array(obj.geometry.index.array);

    const vertexBuffer = {};

    vertexBuffer.position = device.createBuffer({
        label:"obj mesh positions",
        size: positions.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });

        
    device.queue.writeBuffer(vertexBuffer.position, 0, positions);

    vertexBuffer.normal = device.createBuffer({
        label:"obj mesh normals",
        size: normals.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });

    device.queue.writeBuffer(vertexBuffer.normal, 0, normals);

    const indexBuffer = device.createBuffer({
        label:"obj mesh indices",
        size: indices.byteLength,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });

    device.queue.writeBuffer(indexBuffer, 0, indices);

    const uniformBuffer = device.createBuffer({
        size: 4*4*4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const src_shaders = `
        struct VertexOut {
            @builtin(position) position: vec4f,
            @location(${loc_inter_stage_normal}) normal: vec3f
        };

        struct Matrices {
            MVP: mat4x4f,
        };

        @group(${id_group}) @binding(${binding_matrices}) var<uniform> matrices:Matrices;

        @vertex fn main_vert(
            @location(${loc_position}) position: vec3f,
            @location(${loc_normal}) normal: vec3f)
            -> VertexOut {
            var vertex: VertexOut;
            vertex.position = matrices.MVP * vec4f(position, 1);
            vertex.normal = normal;
            return vertex;
        }
        @fragment fn main_frag(@location(${loc_inter_stage_normal}) normal: vec3f)
            -> @location(0) vec4f {
            return vec4f(0.5*(normal + vec3f(1)), 1);
        }
    `;
    const shaderModule = device.createShaderModule({
        label: "solid triangle shader",
        code:src_shaders,
    });

    const pipeline = device.createRenderPipeline({
        label: "solid triangle pipeline",
        layout: "auto",
        vertex: {
            module:shaderModule,
            entryPoint: "main_vert",
            buffers: [
                        {
                            arrayStride: 4*3,
                            attributes: [{
                                format: "float32x3",
                                offset: 0,
                                shaderLocation: loc_position,
                            }],
                        },
                        {
                            arrayStride: 4*3,
                            attributes: [{
                                format: "float32x3",
                                offset: 0,
                                shaderLocation: loc_normal,
                            }],
                        }

                ],
        },
        fragment: {
            module:shaderModule,
            entryPoint: "main_frag",
            targets: [{
                format: preferredFormat
            }]
        },
        primitive:{
            topology: 'triangle-list',
        },
        depthStencil: {
            depthWriteEnabled: true,
            depthCompare: 'less',
            format: format_depth_texture,
        },
    });

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(id_group),
        entries:[
            { binding: binding_matrices, resource: {buffer: uniformBuffer} },
        ],
    });

    function render(renderPass, MVP) {
        renderPass.setPipeline(pipeline);
        device.queue.writeBuffer(uniformBuffer, 0, MVP);
        renderPass.setVertexBuffer(0, vertexBuffer.position);
        renderPass.setVertexBuffer(1, vertexBuffer.normal);
        renderPass.setIndexBuffer(indexBuffer, 'uint32');
        renderPass.setBindGroup(id_group, bindGroup);
        renderPass.drawIndexed(obj.geometry.index.count);
    }

    return {render};
}

// https://github.com/g-truc/glm/blob/master/glm/ext/matrix_projection.inl
function project(p_obj, MVP, viewport)
{
    let tmp = vec4.transformMat4(p_obj, MVP);
    tmp = tmp.map((x) => x/tmp[3]); // tmp /= tmp[3]
    for(let i=0 ; i<2 ; i++) {
        tmp[i] = (0.5*tmp[i] + 0.5) * viewport[i+2] + viewport[i];
    }
    return tmp;
}

// https://github.com/g-truc/glm/blob/master/glm/ext/matrix_projection.inl
function unproject(p_win, MVP, viewport) {
    let MVP_inv = mat4.invert(MVP);
    let tmp = mat4.clone(p_win);

    for (let i = 0; i < 2; i++)
        tmp[i] = 2.0 * (tmp[i] - viewport[i]) / viewport[i+2] - 1.0;

    let p_obj = vec4.transformMat4(tmp, MVP_inv);

    p_obj = p_obj.map((x) => x/p_obj[3]);

    return p_obj;
}

function unproject_vector(vec_win, MVP, viewport)
{
    let org_win = project([0,0,0,1], MVP, viewport);
    let vec = unproject([org_win[0]+vec_win[0], org_win[1]+vec_win[1], org_win[2]+vec_win[2], 1],
                        MVP, viewport);
    return vec;
}




main();