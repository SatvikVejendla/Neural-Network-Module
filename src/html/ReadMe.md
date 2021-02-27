# HTML Quickstart


To get started, create a script such as that follows in order to import the module from static hosting:

```
<script type="module">
        import { Standard } from "https://unpkg.com/neural-network-node@1.2.9/src/html/index.js";
</script>
```


Once you're done with this, you can access the imported classes at any time in any of the script files. HOWEVER, you have to make sure to import the correct class.

For example, if you want to import the deep feed forward network, then use this script instead.

```
<script type="module">
        import { DFF } from "https://unpkg.com/neural-network-node@1.2.9/src/html/index.js";
</script>
```
