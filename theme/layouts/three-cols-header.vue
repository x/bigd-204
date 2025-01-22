<!--
  Usage:
```md
---
layout: three-cols-header
---
This shows up across both
::left::
# Left
This shows on the left
::middle::
# Middle
This shows in the middle
::right::
# Right
This shows on the right
```
-->

<script setup lang="ts">
const props = defineProps({
  class: {
    type: String,
  },
})
</script>

<template>
  <div class="slidev-layout two-columns w-full">
    <div class="col-span-3">
      <slot />
    </div>
    <br>
    <div class="grid grid-cols-3 grid-rows-2">
      <div class="col-left" :class="props.class">
        <slot name="left" />
      </div>
      <div class="col-middle" :class="props.class">
        <slot name="middle" />
      </div>
      <div class="col-right" :class="props.class">
        <slot name="right" />
      </div>
    </div>
  </div>
</template>