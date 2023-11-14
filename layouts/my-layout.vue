<template>
  <div class="cover-container">
    <video class="cover-video" autoplay muted loop>
      <source src="gpt-4-92586ac_1080p60.mp4" type="video/mp4" />
      <!-- 可以放置备用图片 -->
      <!-- <img src="fallback_image.jpg" alt="Fallback image" /> -->
    </video>
    <slot></slot> <!-- 允许插入其他内容 -->
  </div>
</template>

<script setup>
// Vue 代码逻辑可以放这里
</script>

<style scoped>
.cover-container {
  position: relative;
  width: 100%;
  height: 100%;
}

.cover-video {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
  z-index: -1;
}
</style>