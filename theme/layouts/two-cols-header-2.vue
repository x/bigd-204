<!--
  Usage:
```md
---
layout: two-cols-header-2
---
This shows up across both
::left::
# Left
This shows on the left
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
    <div class="col-span-2">
      <slot />
    </div>
    <div class="grid grid-cols-2 grid-rows-2">
      <div class="col-left" :class="props.class">
        <slot name="left" />
      </div>
      <div class="col-right" :class="props.class">
        <slot name="right" />
      </div>
    </div>
  </div>
</template>