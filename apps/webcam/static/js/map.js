const beachcams = [];

function map_add_beachcam(bc) {
    beachcams.push(bc);
}

function map_make() {
    const map = L.map('map');
    const bounds = L.latLngBounds(L.latLng(40.15, 0.77), L.latLng(38.55, 4.85));
    map.fitBounds(bounds);
    map.setMaxBounds(bounds);
    
    L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }).addTo(map);

    let current = null;

    for (const m of beachcams) {
        const p = m.max_crowd_count && m.last_prediction && (m.last_prediction / m.max_crowd_count) || 0.5;
        let currentClass = '';
        if (m.current) {
            currentClass = ' current';
            current = m;
        }
        
        const marker = L.marker([m.latitude, m.longitude], {
            icon: L.divIcon({
                iconSize: "auto",
                html: `<div class="inner${currentClass}" style="--p: ${p}">${Math.round(m.last_prediction)}</div>`,
                className: 'map_prediction_icon',
                iconSize: [32, 32]
            })
        }).addTo(map);

        marker.on('click', (e) => {
            window.location.href = '/platja/' + m.slug;
        });
    }

    if (current) {
        const marker = L.marker([current.latitude, current.longitude]).addTo(map);

        marker.on('click', (e) => {
            window.location.href = '/platja/' + current.slug;
        });
    }
}