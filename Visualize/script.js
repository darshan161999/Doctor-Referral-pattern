// Set the dimensions and margins of the diagram
const margin = { top: 20, right: 90, bottom: 30, left: 90 },
    width = 960 - margin.left - margin.right,
    height = 800 - margin.top - margin.bottom;

// Append the svg object to the body of the page
const svg = d3.select("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

let i = 0,
    duration = 750,
    root;

// Declares a tree layout and assigns the size
const treemap = d3.tree().size([height, width]);

// Load the external data
d3.json("fhir_resources.json").then(function(data) {
    root = d3.hierarchy(formatData(data), d => d.children);
    root.x0 = height / 2;
    root.y0 = 0;
  
    // Collapse after the second level
    root.children.forEach(collapse);

    update(root);
});

function formatData(data) {
    const formatted = { name: "root", children: [] };
    data.Patient.forEach(patient => {
        const patientNode = {
            name: patient.resource.id,
            children: data.Encounter.filter(e => e.resource.subject.reference === `Patient/${patient.resource.id}`)
                .map(encounter => {
                    return { name: encounter.resource.id };
                })
        };
        if (patientNode.children.length) {
            formatted.children.push(patientNode);
        }
    });
    return formatted;
}

function collapse(d) {
    if (d.children) {
        d._children = d.children;
        d._children.forEach(collapse);
        d.children = null;
    }
}

function update(source) {
    // Code to create and update nodes and links
    // This is similar to the example provided earlier in the previous response
}

// Function to create a curved (diagonal) path from parent to the child nodes
function diagonal(s, d) {
    const path = `M ${s.y} ${s.x}
            C ${(s.y + d.y) / 2} ${s.x},
              ${(s.y + d.y) / 2} ${d.x},
              ${d.y} ${d.x}`;
    return path;
}

// Toggle children on click.
function click(event, d) {
    if (d.children) {
        d._children = d.children;
        d.children = null;
    } else {
        d.children = d._children;
        d._children = null;
    }
    update(d);
}
