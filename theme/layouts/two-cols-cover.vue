<!--
  Usage:
```md
---
layout: two-cols-cover
---
# Left
This shows on the left (vertically centered)
::right::
# Right
This shows on the right (vertically centered)
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
  <div class="slidev-layout two-columns w-full cover">
    <div class="grid grid-cols-2">
      <div class="col-left my-auto" :class="props.class">
        <slot />
      </div>
      <div class="col-right my-auto" :class="props.class">
        <slot name="right" />
      </div>
    </div>
  </div>
</template>