<!--
  Usage:
```md
---
layout: two-cols-two-rows-header
---
This shows up across both
::left-top::
# Left
This shows on the left top

::left-bottom::
This shows on the left bottom

::right-top::
# Right
This shows on the right top

::right-bottom::
This shows on the right bottom
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
    <br>
    <div class="grid grid-cols-2 grid-rows-2">
      <div class="col-left" :class="props.class">
        <slot name="left-top" />
      </div>
      <div class="col-right" :class="props.class">
        <slot name="right-top" />
      </div>
      <div class="col-left" :class="props.class">
        <slot name="left-bottom" />
      </div>
      <div class="col-right" :class="props.class">
        <slot name="right-bottom" />
      </div>
    </div>
  </div>
</template>