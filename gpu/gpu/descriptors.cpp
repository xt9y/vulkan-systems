#include "descriptors.hpp"
#include <map>
#include <stdexcept>

namespace gpu
{
    void descriptors::create(std::vector<vk::DescriptorSetLayoutBinding> bindings,
                             std::uint32_t count)
    {
        if(layout_)throw std::logic_error("descriptors already created");
        if(!count || bindings.empty())throw std::invalid_argument("empty descriptor layout");
        std::map<vk::DescriptorType,std::uint32_t> counts;
        std::map<std::uint32_t,bool> slots;
        for(const auto& b:bindings)
        {
            if(!b.descriptorCount || !b.stageFlags || slots[b.binding])
                throw std::invalid_argument("invalid/duplicate descriptor binding");
            slots[b.binding]=true;
            counts[b.descriptorType]+=b.descriptorCount*count;
        }
        const auto device=vk::device::get()->handle;
        try
        {
            vk::DescriptorSetLayoutCreateInfo info{};
            info.bindingCount=std::uint32_t(bindings.size());
            info.pBindings=bindings.data();
            layout_=vk::descriptor_layout::create(info);
            std::vector<vk::DescriptorPoolSize> sizes;
            for(const auto& entry:counts)
                sizes.emplace_back(entry.first,entry.second);
            vk::DescriptorPoolCreateInfo pool_info{};
            pool_info.maxSets=count;
            pool_info.poolSizeCount=std::uint32_t(sizes.size());
            pool_info.pPoolSizes=sizes.data();
            pool_=device.createDescriptorPool(pool_info);
            std::vector<vk::DescriptorSetLayout> layouts(count,layout_);
            vk::DescriptorSetAllocateInfo allocation{};
            allocation.descriptorPool=pool_;
            allocation.descriptorSetCount=count;
            allocation.pSetLayouts=layouts.data();
            sets_=device.allocateDescriptorSets(allocation);
            bindings_=std::move(bindings);
        }
        catch(...)
        {
            reset();
            throw;
        }
    }

    void descriptors::reset()
    {
        const auto device=vk::device::get()->handle;
        sets_.clear();
        if(pool_)device.destroyDescriptorPool(pool_);
        pool_=vk::DescriptorPool{};
        if(layout_)vk::descriptor_layout::destroy(layout_);
        layout_=vk::DescriptorSetLayout{};
        bindings_.clear();
    }

    vk::DescriptorSet descriptors::set(std::uint32_t index) const
    {
        if(index>=sets_.size())throw std::out_of_range("descriptor set index");
        return sets_[index];
    }

    const vk::DescriptorSetLayoutBinding& descriptors::lookup(
        std::uint32_t binding,std::uint32_t element) const
    {
        for(const auto& b:bindings_)
            if(b.binding==binding)
            {
                if(element>=b.descriptorCount)
                    throw std::out_of_range("descriptor array element");
                return b;
            }
        throw std::out_of_range("descriptor binding");
    }

    void descriptors::write_buffer(std::uint32_t index,std::uint32_t binding,
                                   vk::DescriptorBufferInfo info,std::uint32_t element)
    {
        const auto& b=lookup(binding,element);
        if(b.descriptorType!=vk::DescriptorType::eUniformBuffer &&
           b.descriptorType!=vk::DescriptorType::eStorageBuffer &&
           b.descriptorType!=vk::DescriptorType::eUniformBufferDynamic &&
           b.descriptorType!=vk::DescriptorType::eStorageBufferDynamic)
            throw std::invalid_argument("binding is not a buffer descriptor");
        vk::WriteDescriptorSet write{};
        write.dstSet=set(index);
        write.dstBinding=binding;
        write.dstArrayElement=element;
        write.descriptorType=b.descriptorType;
        write.descriptorCount=1;
        write.pBufferInfo=&info;
        vk::device::get()->handle.updateDescriptorSets({write},{});
    }

    void descriptors::write_image(std::uint32_t index,std::uint32_t binding,
                                  vk::DescriptorImageInfo info,std::uint32_t element)
    {
        const auto& b=lookup(binding,element);
        if(b.descriptorType!=vk::DescriptorType::eStorageImage &&
           b.descriptorType!=vk::DescriptorType::eSampledImage &&
           b.descriptorType!=vk::DescriptorType::eCombinedImageSampler &&
           b.descriptorType!=vk::DescriptorType::eSampler &&
           b.descriptorType!=vk::DescriptorType::eInputAttachment)
            throw std::invalid_argument("binding is not an image/sampler descriptor");
        vk::WriteDescriptorSet write{};
        write.dstSet=set(index);
        write.dstBinding=binding;
        write.dstArrayElement=element;
        write.descriptorType=b.descriptorType;
        write.descriptorCount=1;
        write.pImageInfo=&info;
        vk::device::get()->handle.updateDescriptorSets({write},{});
    }
}
